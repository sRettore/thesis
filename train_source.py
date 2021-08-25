import torch
from torch import distributed
import torch.nn.functional as F

import os
import time
from PIL import Image
from utils.run_utils import *
import numpy as np
from losses.loss_source_handler import LossSourceHandler
from losses.loss_target_handler import LossTargetHandler
from utils import get_regularizer
from functools import reduce

class Trainer:
    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, logdir=None):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.where_to_sim = opts.where_to_sim
        self.step = opts.step
        self.no_mask = opts.no_mask  # if True sequential dataset from https://arxiv.org/abs/1907.13372
        self.overlap = opts.overlap
        self.loss_de_prototypes_sumafter = opts.loss_de_prototypes_sumafter
        self.num_classes = sum(classes) if classes is not None else 0
        self.multi = opts.multi # multi level loss
        self.delta = opts.multi_delta # delta for ensemble of probabilities
        self.freezeBn = opts.fix_bn
        
        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0
            
        args_uda = {
            'lce': opts.uda_lce,
            'lsce': opts.uda_lsce,
            'lIWsce': opts.uda_lIWsce,
            'lmsq': opts.uda_lmsq,
            'lIWmsq': opts.uda_lIWmsq,
            'lkd': opts.uda_lkd,
            'lkd_alpha': opts.uda_lkd_alpha
            }

        args_low = {
            'lce': opts.low_lce,
            'lsce': opts.low_lsce,
            'lIWsce': opts.low_lIWsce,
            'lmsq': opts.low_lmsq,
            'lIWmsq': opts.low_lIWmsq,
            'lkd': False,
            'lkd_alpha': 1.
            }

        # Loss Handlers
        self.lossSource = LossSourceHandler(device, opts, has_model_old = (model_old is not None), num_classes = self.num_classes, old_classes = self.old_classes, logdir = logdir)
        
        if opts.multi:
            self.lossLowUDA = LossTargetHandler(args_low, applySoftMax=False)
        
        # Regularization Loss (EWC, RW, PI)
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance
        
        #require features
        self.ret_intermediate = self.lossSource.require_intermediate()

    def train(self, world_size, cur_epoch, iterations, optim, source_loader, target_loader, scheduler=None, print_int=10, logger=None,
              prototypes=None, count_features=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch + 1, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        self.lossSource.reset_dict()
        lreg = torch.tensor(0.)
        
        source_loader.sampler.set_epoch(cur_epoch)

        if self.freezeBn:
            model.eval()
        else:
            model.train()
            
        def batch_generator(source_loader, iterations):     
            source_iter = iter(source_loader)           
            for i in range (iterations):
                yield i, next(source_iter)
                
        start_time = time.time()
        start_epoch_time = time.time()
       
        # We suppose that dataset always return a label object, in case it's missing it return a None object
        for cur_step, (source_images, source_labels, indices) in batch_generator(source_loader, iterations):
            features_old = None    
            features = None
            
            ##########################
            # source supervised loss #
            ##########################
            # train with source
            images = source_images.to(device) #, dtype=torch.float32)
            labels = source_labels.to(device, dtype=torch.long)
            
            if self.lossSource.require_model_old() and self.model_old is not None:
                with torch.no_grad():
                    outputs_old = self.model_old(images, self.ret_intermediate)
                    if isinstance(outputs_old, tuple):
                        outputs_old = outputs_old[0]
                        if self.ret_intermediate:
                            features_old = outputs_old[2] 
            else:
                outputs_old = None

            optim.zero_grad()
            
            outputs = model(images, ret_intermediate=self.ret_intermediate)   
            if isinstance( outputs, tuple):
                outputs = outputs[0]
                if self.ret_intermediate:
                    features = outputs[2]    

            if self.lossSource.require_running_stats():
                prototypes, count_features = self._update_running_stats((F.interpolate(
                    input=labels.unsqueeze(dim=1).double(), size=(features['body'].shape[2], features['body'].shape[3]),
                    mode='nearest')).long(), features['body'], self.no_mask, self.overlap, self.step, prototypes,
                                                                        count_features)
            else:
                prototypes = None
                count_features = None
            
            loss_tot_source = self.lossSource.train_batch(cur_epoch, cur_step, labels, 
                                                   outputs, features, 
                                                   outputs_old, features_old, 
                                                   prototypes, count_features)   
            if self.where_to_sim == 'GPU_server':
                from apex import amp
                with amp.scale_loss(loss_tot_source, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_tot_source.backward()
            
            ###############
            # reg. loss   #
            ###############
            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows' or distributed.get_rank() == 0:
                    self.regularizer.update()
                lreg = self.reg_importance * self.regularizer.penalty()
                if lreg != 0.:
                    if self.where_to_sim == 'GPU_server':
                        with amp.scale_loss(lreg, optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        lreg.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            ####################
            # metrics & Output #
            ####################
            
            epoch_loss, reg_loss, interval_loss = self.lossSource.output_values(epoch_loss, reg_loss, interval_loss)         

            reg_loss += lreg.item() if lreg.item() != 0. else 0.
            interval_loss += lreg.item() if lreg.item() != 0. else 0.
                    
            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch + 1}, Batch {cur_step + 1}/{iterations},"
                            f" Loss={interval_loss:0.9f}, Time taken={(time.time() - start_time):.6f}")
                logger.debug("Loss made of:   "+self.lossSource.lossText()+(f",   LREG: {lreg}" if self.regularizer_flag else ""))
                
                # visualization
                if logger is not None:
                    x = cur_epoch * iterations + cur_step + 1
                    logger.add_scalar('Losses/interval_loss', interval_loss, x)
                    self.lossSource.addToLogger(logger, x)
                    
                    if self.regularizer_flag:
                        logger.add_scalar('Losses/lreg', lreg.item(), x)


                interval_loss = 0.0
                start_time = time.time()

        logger.info(f"END OF EPOCH {cur_epoch + 1}, TOTAL TIME={(time.time() - start_epoch_time):.6f}")

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(device)
        reg_loss = torch.tensor(reg_loss).to(device)

        if not self.where_to_sim == 'GPU_windows':
            torch.distributed.reduce(epoch_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

        if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
            epoch_loss = epoch_loss / world_size / iterations
            reg_loss = reg_loss / world_size / iterations
        else:
            if distributed.get_rank() == 0:
                epoch_loss = epoch_loss / distributed.get_world_size() / iterations
                reg_loss = reg_loss / distributed.get_world_size() / iterations

        logger.info(f"Epoch {cur_epoch + 1}, Class Loss={epoch_loss:.9f}, Reg Loss={reg_loss:.9f}")
        #print("Class Counter: ", self.classCounters)
        return (epoch_loss, reg_loss), prototypes, count_features

    def validate(self, loader, metrics, world_size, ret_samples_ids=None, logger=None, vis_dir=None, label2color=None, denorm=None, print_int=0, epoch=0, valTarget=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lreg = torch.tensor(0.)
        self.lossSource.reset_dict()
     
        iterations = len(loader)
        
        mean_p = torch.tensor(0.)
        list_mean = torch.zeros(255)
        #self.classCounters = [0] * self.num_classes
            
        ret_samples = []
        with torch.no_grad():
            for cur_step, (images, labels, indices) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)
                """
                for n in torch.unique(labels):
                    if n < len (self.classCounters):
                        self.classCounters[n] += 1  
                """
                features_old = None  
                features = None
                
                if self.lossSource.require_model_old() and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old = self.model_old(images, self.ret_intermediate)
                        if isinstance(outputs_old, tuple):
                            outputs_old = outputs_old[0]
                            if self.ret_intermediate:
                                features_old = outputs_old[2] 
                else:
                    outputs_old = None   
                    
                    
                outputs = model(images, self.ret_intermediate)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    if self.ret_intermediate:
                        features = outputs[2] 
                
                # validate source
                self.lossSource.validate_batch(cur_step, labels, outputs, features, outputs_old, features_old)
                class_loss, reg_loss, interval_loss = self.lossSource.output_values(class_loss, reg_loss, interval_loss)
                
                # validate Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    lreg  = self.reg_importance * self.regularizer.penalty()
                    reg_loss += lreg.item() if lreg.item() != 0. else 0.
                    interval_loss += lreg.item() if lreg.item() != 0. else 0.            

                #softmax = torch.softmax(outputs, dim=1)
                #mean_p = mean_p + torch.mean(softmax, dim=(0,2,3))
                _, prediction = outputs.max(dim=1)
                #argmax, _ = softmax.max(dim=1)
                
                #for n in torch.unique(prediction):
                #    list_mean[n] = list_mean[n] + torch.mean(argmax[prediction == n])                    
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if print_int >0 and (cur_step + 1) % print_int == 0:
                    interval_loss = interval_loss / print_int
                    if logger is not None:
                        logger.info(f"Validation, Batch {cur_step + 1}/{iterations},"
                                    f" Loss={interval_loss}")
                        logger.debug("Loss made of:   "+self.lossSource.lossText()+(f", lreg: {lreg}" if self.regularizer_flag else ""))
                        #print("Class Counter: ", self.classCounters)
                    else:
                        print(f"Validation, Batch {cur_step + 1}/{iterations},"
                                    f" Loss={interval_loss}")
                        print("Loss made of:   "+self.lossSource.lossText()+(f", lreg: {lreg}" if self.regularizer_flag else ""))
                        #print("Class Counter: ", self.classCounters)
                    interval_loss = 0.0   
   
                if vis_dir is not None and denorm is not None and label2color is not None and ret_samples_ids is None:
                    image_name = loader.dataset.dataset.dataset.getFilename(indices[0])
                    image_tosave, label_tosave, _ = loader.dataset.dataset.dataset[indices[0]]
                    image_tosave = (denorm(image_tosave).numpy() *255).astype(np.uint8).transpose(1,2,0)
                    prediction_tosave = label2color(prediction)[0].astype(np.uint8)
                    label_tosave = label2color(labels)[0].astype(np.uint8)
                    
                    Image.fromarray(image_tosave).save(f'{vis_dir}/test{"_target" if valTarget else ""}_{indices[0]}_RGB.jpg')
                    Image.fromarray(prediction_tosave).save(f'{vis_dir}/test{"_target" if valTarget else ""}_{indices[0]}_pred.png')
                    Image.fromarray(label_tosave).save(f'{vis_dir}/test{"_target" if valTarget else ""}_{indices[0]}_label.png')
                
                if ret_samples_ids is not None and denorm is not None and label2color is not None and cur_step in ret_samples_ids:  # get samples
                    image_tosave, label_tosave, _ = loader.dataset.dataset.dataset[indices[0]]
                    image_tosave = (denorm(image_tosave).numpy() *255).astype(np.uint8).transpose(1,2,0)
                    prediction_tosave = label2color(prediction)[0].astype(np.uint8)
                    label_tosave = label2color(labels)[0].astype(np.uint8)
                    ret_samples.append((image_tosave, prediction_tosave, label_tosave))
                    
                    if vis_dir is not None:
                        Image.fromarray(image_tosave).save(f'{vis_dir}/epoch{epoch}{"_target" if valTarget else ""}_{indices[0]}_RGB.jpg')
                        Image.fromarray(prediction_tosave).save(f'{vis_dir}/epoch{epoch}{"_target" if valTarget else ""}_{indices[0]}_pred.png')
                        Image.fromarray(label_tosave).save(f'{vis_dir}/epoch{epoch}{"_target" if valTarget else ""}_{indices[0]}_label.png')                           
                                        
            #print(f"Mean p = {mean_p/iterations},\n  list_mean = {list_mean/iterations}")
            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            if not self.where_to_sim == 'GPU_windows':
                torch.distributed.reduce(class_loss, dst=0)
                torch.distributed.reduce(reg_loss, dst=0)
                
            if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                class_loss = class_loss / world_size / iterations
                reg_loss = reg_loss / world_size / iterations

            else:
                if distributed.get_rank() == 0:
                    class_loss = class_loss / distributed.get_world_size() / iterations
                    reg_loss = reg_loss / distributed.get_world_size() / iterations

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss:.9f}, Reg Loss={reg_loss:.9f} (without scaling)")

        #print("Final Class Counter: ", self.classCounters)                
        return (class_loss, reg_loss), score, ret_samples
    
    def state_dict(self):
        state = {'regularizer': self.regularizer.state_dict() if self.regularizer_flag else None}
        return state

    def load_state_dict(self, state):
        if state['regularizer'] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state['regularizer'])

    def _update_running_stats(self, labels_down, features, sequential, overlapped, incremental_step, prototypes, count_features):
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if overlapped or ((not sequential) and incremental_step > 0):
            cl_present = cl_present[1:]

        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        features_local_mean = torch.zeros([self.num_classes, 2048], device=self.device)

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1],-1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features




