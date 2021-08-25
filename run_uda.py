import utils
import argparser

import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils import data
from shutil import copy
import time

import torch
torch.backends.cudnn.benchmark = True

from utils.run_utils import *
from metrics import StreamSegMetrics

from utils import train_helper
from train_source import Trainer
from train_target import TrainerUDA
import tasks
from utils import BatchGeneratorSkipping
from utils import DistributedWeightedSampler


def main(opts):
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{task_name}-{opts.name}/"

    device, rank, world_size, logger = define_distrib_training(opts, logdir_full)

    logger.print("[!] run UDA PYTHON!!!!")

    ###################
    # WARNING         #
    ###################
    
    if opts.uda_loss and not opts.uda_target:
        logger.info("[!] WARNING: Domain Adaptation is activated but no target dataset is selected, it will be ignored")
       
    if not (opts.uda_loss or opts.multi) and opts.uda_target:
        logger.info(f"[!] WARNING: Target Loss is not configured but a target dataset is selected, test will be done on {'target' if opts.uda_validate_target else 'source'} dataset")

    if opts.low_loss and not opts.multiDeeplab:
        logger.info("[!] WARNING: Multi-level guidance loss is configured but Multi-level guidance is not activated, it will be ignored")
  
    if opts.multiDeeplab and not opts.uda_target:
        opts.multiDeeplab = False
        opts.multi = False
        logger.info("[!] WARNING: Multi-level guidance is activated but no target dataset is selected, it will be ignored")

    if not opts.low_loss and opts.multiDeeplab:   
        opts.multiDeeplab = False
        opts.multi = False
        logger.info("[!] WARNING: Multi-level guidance loss is configured but Multi-level guidance is activated, it will be ignored")

    logger.print(f"Device: {device}")
    logger.print(f"Rank: {rank}, world size: {world_size}")

    # Set up random seed
    setup_random_seeds(opts)

    ###################
    # SET DATASET     #
    ################### 
    
    source_dst, target_dst, val_dst, test_dst, n_classes = get_dataset(opts, rank=rank)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    if opts.weight_samples > 0:
        source_loader = data.DataLoader(source_dst, batch_size=opts.batch_size,
                                       sampler=DistributedWeightedSampler(source_dst.weights, len(source_dst), replacement=False, num_replicas=world_size, rank=rank),
                                       num_workers=opts.num_workers, drop_last=True)
    else:
        source_loader = data.DataLoader(source_dst, batch_size=opts.batch_size,
                                       sampler=DistributedSampler(source_dst, num_replicas=world_size, rank=rank),
                                       num_workers=opts.num_workers, drop_last=True)    
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)   
    if opts.uda_target:
        target_loader = data.DataLoader(target_dst, batch_size=opts.target_batch_size,
                                   sampler=DistributedSampler(target_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    else:
        target_loader = None
                            
    test_batch_size = opts.target_batch_size if opts.uda_validate_target else opts.batch_size
    test_loader = data.DataLoader(test_dst, batch_size=test_batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    ###################
    # SET ITERATIONS  #
    ###################                              
                                  
    source_len = len(source_dst)      
    iterations = len(source_loader)
    
    ##if iterations > 5000:
    ##   iterations = 5000
    
    if opts.uda_target:
        target_len = len(target_dst)
        if len(target_loader) < len(source_loader):
            iterations = len(target_loader)
            
            ##if iterations > 5000:
            ##    iterations = 5000
            
        target_len = iterations*opts.target_batch_size if iterations*opts.target_batch_size < target_len else target_len
    source_len = iterations*opts.batch_size if iterations*opts.batch_size < source_len else source_len
       
    if not opts.uda_target:
        logger.info(f"Dataset: {opts.dataset}, n.Classes {n_classes}")
        logger.info(f"Train set: {source_len}, Val set: {len(val_dst)}, Test set: {len(test_dst)}")
        logger.info(f"Total batch size is {opts.batch_size * world_size}")
    else:
        logger.info(f"Source Dataset: {opts.dataset}, Target Dataset: {opts.target_dataset}, n.Classes: {n_classes}")
        logger.info(f"Train set: {source_len} ({len(source_dst)}), Target set: {target_len} ({len(target_dst)}), Val set: {len(val_dst)}, "
                    f"Test Dataset: {len(test_dst)} {'(source)' if opts.uda_validate_source else ''}")
        logger.info(f"Total batch size is {opts.total_batch_size * world_size} ({opts.batch_size * world_size}+{opts.target_batch_size * world_size})")

    ###################
    # SET MODELS      #
    ###################       
    #logger.info(f"Backbone: {opts.backbone}")

    step_checkpoint = None
    model, model_params = train_helper.get_model(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step), opts)

    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old, _ = train_helper.get_model(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step-1), opts)

    if opts.fix_bn:
        model.eval()

    #logger.debug(model)

    ###################
    # SET OPTIMIZER   #
    ###################
    
    params = []
    for i in range (len(model_params)):
        if i != 0 or (i == 0 and not opts.freeze):
            params.append(model_params[i])
            
    """        
    if not opts.freeze:
        params.append(model_params[0])
    params.append(model_params[1])      
    """

    optimizer = torch.optim.SGD(params, weight_decay=opts.weight_decay, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * iterations, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    ###################
    # MODEL TO DEVICE #
    ###################
    
    if model_old is not None:
        if opts.where_to_sim == 'GPU_server':
            from apex.parallel import DistributedDataParallel
            from apex import amp
            [model, model_old], optimizer = amp.initialize([model.to(device), model_old.to(device)], optimizer,
                                                     opt_level=opts.opt_level)
            model_old = DistributedDataParallel(model_old)
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)
            model_old = model_old.to(device)
    else:
        if opts.where_to_sim == 'GPU_server':
            from apex.parallel import DistributedDataParallel
            from apex import amp
            model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)
            # Put the model on GPU
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)

    ###################
    # LOAD OLD MODEL  #
    ###################
    
    # Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path       
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            if opts.continue_with_round:
                path = f"{logdir_full}/{task_name}-{opts.name}_{opts.step -1}_r1_backup{opts.backup}.pth"    
            else:
                path = f"{logdir_full}/{task_name}-{opts.name}_{opts.step -1}_backup{opts.backup}.pth"    
        
        
        if opts.step_ckpt is not None:
            path_FT = opts.step_ckpt
        else:
            path_FT = path.replace(opts.name, 'FT')
            
        if (not os.path.exists(path)):  # and opts.name != 'EWC' and opts.name != 'MiB' and opts.name != 'PI' and opts.name != 'RW':
            logger.info(f"{path} not exists")
            path = path_FT
            if (os.path.exists(path)):
                logger.info(f"[!] WARNING: Step Checkpoint {path}, using FT instead: {path_FT}")
            else:
                if opts.continue_with_round:
                    path_FT = f"{opts.logdir}//{task_name}//{task_name}-FT//{task_name}-FT_{opts.step - 1}_r1_backup{opts.backup}.pth"
                else:
                    path_FT = f"{opts.logdir}//{task_name}//{task_name}-FT//{task_name}-FT_{opts.step - 1}_backup{opts.backup}.pth"
                path = path_FT
                logger.info(f"[!] WARNING: FT step Checkpoint not found, looking for default checkpoint: {path_FT}")
             	
        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            load_ckpt(step_checkpoint, model, strict=False)

            if opts.init_balanced:
                # implement the balanced initialization for the new step (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.init_new_classifier(device)
                
            # Load state dict from the model state dict, that contains the old model 
            if opts.step > 0:
                load_ckpt(step_checkpoint, model_old, strict=False)
            
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
            #exit()
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        if opts.step > 0:
            for par in model_old.parameters():
                par.requires_grad = False
            model_old.eval()

    ###################
    # SET TRAINER     #
    ###################        

    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 or opts.round > 1 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    if opts.uda or (opts.target_dataset and opts.test):
        trainerClass = TrainerUDA
    else:
        trainerClass = Trainer

    trainer = trainerClass(model, model_old, device=device, opts=opts, trainer_state=trainer_state,
                         classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step), logdir=logdir_full)



    ###################
    # LOAD CHECKPOINT #
    ###################     
    # Handle checkpoint for current model (model old will always be as previous step or None)
    
    if opts.round > 0:
        opts.ckpt = f"{logdir_full}/{task_name}-{opts.name}_{opts.step}_backup{opts.backup}.pth"    
        logger.info(f"[!] Selecting Round Checkpoint: {opts.ckpt}")
        
    best_score = 0.0
    cur_epoch = 0    
    if opts.test and opts.ckpt is None:
        opts.ckpt = f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}_backup{opts.backup}.pth"
        
    if opts.test and not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(opts.ckpt)
        
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        load_ckpt(checkpoint, model, strict=True)
        logger.info("[!] Model restored from %s" % opts.ckpt)
        
        if opts.ckpt_resume_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epoch = checkpoint["epoch"] + 1
            best_score = checkpoint['best_score']
            logger.info("[!] Optimizer and Trainer restored to resume training")
            # if we want to resume training, resume trainer from checkpoint
            if 'trainer_state' in checkpoint:
                trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0 and not opts.test:
            logger.info(f"[!] Train from scratch {' for the new round' if opts.round > 0 else ''}")

    ###################
    # TRAINING SETUP  #
    ###################      
    
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        #sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        sample_ids = np.random.choice(len(test_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    # convert labels to images
    label2color = utils.Label2Color(cmap=utils.color_map(opts.target_dataset if opts.uda_validate_target else opts.dataset , test_dst.inverted_order))  # convert labels to images

    denorm = utils.get_denorm_transform()
    denorm2 = utils.get_denorm_transform()
    
    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes, opts, test_dst.labelsName())
    results = {}

    # check if random is equal here.
    logger.print(f"Random check: {torch.randint(0,100, (1,1)).item() }")

    # load prototypes if needed
    logger.info(f"Prototypes initialization to zero vectors")
    prototypes = torch.zeros([sum(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)),
                              opts.feat_dim])
    prototypes.requires_grad = False
    count_features = torch.zeros([sum(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))],
                                 dtype=torch.long)
    count_features.requires_grad = False
    if opts.step > 0 and (opts.loss_de_prototypes > 0 or opts.lfc_sep_clust > 0 or opts.loss_fc):
        logger.info(f"Prototypes loaded from previous checkpoint")

        prototypes_old = step_checkpoint["prototypes"]
        count_features_old = step_checkpoint["count_features"]

        prototypes[0:prototypes_old.shape[0],:] = prototypes_old
        count_features[0:count_features_old.shape[0]] = count_features_old
        del step_checkpoint

        logger.info(f"Current prototypes are {prototypes}")
        logger.info(f"Current count_features is {count_features}")
    prototypes = prototypes.to(device)
    count_features = count_features.to(device)

    #gen = BatchGeneratorSkipping(source_loader, target_loader, opts.epochs, iterations, starting_epoch=cur_epoch)
   
    ###################
    # TRAINING        #
    ###################   

    vis_dir_t = os.path.join(logdir_full,f"test-pics-step{opts.step}-backup{opts.backup}-ret_samples-Target")
    os.makedirs(vis_dir_t, exist_ok=True)

    vis_dir = os.path.join(logdir_full,f"test-pics-step{opts.step}-backup{opts.backup}-ret_samples-Source")
    os.makedirs(vis_dir, exist_ok=True)
    
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        """
        with generator
        epoch_loss, prototypes, count_features = trainer.train(cur_epoch=cur_epoch, iterations = iterations, batch_generator=gen, optim=optimizer, world_size=world_size,
                                   source_loader=source_loader, target_loader = target_loader, scheduler=scheduler, logger=logger,
                                   print_int=opts.print_interval, prototypes=prototypes, count_features=count_features)
        """
        epoch_loss, prototypes, count_features = trainer.train(cur_epoch=cur_epoch, iterations = iterations, optim=optimizer, world_size=world_size,
                                   source_loader=source_loader, target_loader = target_loader, scheduler=scheduler, logger=logger,
                                   print_int=opts.print_interval, prototypes=prototypes, count_features=count_features)
        logger.info(f"End of Epoch {cur_epoch+1}/{opts.epochs}, Average Loss={(epoch_loss[0]+epoch_loss[1]):.9f},"
                    f" Class Loss={(epoch_loss[0]):.9f}, Reg Loss={(epoch_loss[1]):.9f}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0]+epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        
        # best model to build incremental steps
        save_ckpt(f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}_val_best.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score, prototypes, count_features)
        logger.info("[!] New Best Checkpoint saved.")

        if ((cur_epoch + 1) % opts.val_interval == 0) or (cur_epoch+1) == opts.epochs: # >= (opts.epochs -1):#== opts.epochs:
            model.eval()
            logger.info("Validate on test set...")        
            t_loss, t_score, t_samples = trainer.validate(loader=test_loader, metrics=val_metrics,
                                                      world_size=world_size, ret_samples_ids=sample_ids, logger=logger, label2color=label2color, denorm=denorm2, print_int=opts.val_print_interval, vis_dir=vis_dir_t, epoch = cur_epoch, valTarget=True)

            # print final parameters
            logger.print("Done test")
            logger.info(f"*** End of Test, Total Loss={t_loss[0]+t_loss[1]},"
                        f" Class Loss={t_loss[0]}, Reg Loss={t_loss[1]}")
            logger.info(val_metrics.to_str(t_score)) 

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("VT-Loss", t_loss[0]+t_loss[1], cur_epoch)
            logger.add_scalar("VT-Loss-reg", t_loss[1], cur_epoch)
            logger.add_scalar("VT-Loss-cls", t_loss[0], cur_epoch)
            logger.add_scalar("VT_Overall_Acc", t_score['Overall Acc'], cur_epoch)
            logger.add_scalar("VT_MeanIoU", t_score['Mean IoU'], cur_epoch)
            logger.add_table("VT_Class_IoU", t_score['Class IoU'], cur_epoch)
            logger.add_table("VT_Acc_IoU", t_score['Class Acc'], cur_epoch)
            logger.add_table("VT_Prc_IoU", t_score['Class Prc'], cur_epoch)
            # logger.add_figure("ValT_Confusion_Matrix", t_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["VT-IoU"] = t_score['Class IoU']
            results["VT-Acc"] = t_score['Class Acc']
            results["VT-Prc"] = t_score['Class Prc']

            for k, (img, target, lbl) in enumerate(t_samples):
                concat_img = np.concatenate((img, lbl, target), axis=1)  # concat along width
                Image.fromarray(concat_img).save(f'{vis_dir_t}/concatT{cur_epoch}_{k}.jpg')
                logger.add_image(f'SampleT_{k}', concat_img.transpose(2,0, 1), cur_epoch)
            
            logger.info("Validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics, world_size=world_size,
                                                                ret_samples_ids=sample_ids, logger=logger, label2color=label2color, denorm=denorm, print_int=opts.val_print_interval, vis_dir=vis_dir, epoch = cur_epoch)
            #val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics, world_size=world_size,
            #                                                    ret_samples_ids=sample_ids, logger=logger, label2color=label2color, denorm=denorm, print_int=opts.val_print_interval)
            logger.print("Done validation on Val set")
            logger.info(f"End of Validation {cur_epoch+1}/{opts.epochs}, Validation Loss={(val_loss[0]+val_loss[1]):.9f},"
                        f" Class Loss={(val_loss[0]):.9f}, Reg Loss={(val_loss[1]):.9f}")
            logger.info(val_metrics.to_str(val_score))
            
            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                
                if score > best_score:
                    best_score = score
                    # best model to build incremental steps
                    save_ckpt(f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}_val_best.pth",
                              model, trainer, optimizer, scheduler, cur_epoch, score, prototypes, count_features)
                    logger.info("[!] New Best Checkpoint saved.")
                else:
                    logger.info("[!] The MIoU of val does not improve, no checkpoint is saved")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0]+val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("V-Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("V-MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            logger.add_table("Val_Prc_IoU", val_score['Class Prc'], cur_epoch)
            # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']
            results["V-Prc"] = val_score['Class Prc']

            for k, (img, target, lbl) in enumerate(ret_samples):
                concat_img = np.concatenate((img, lbl, target), axis=1)  # concat along width
                Image.fromarray(concat_img).save(f'{vis_dir}/concat{cur_epoch}_{k}.jpg')
                logger.add_image(f'Sample_{k}', concat_img.transpose(2,0, 1), cur_epoch)
                
        cur_epoch += 1

    ###################
    # SAVE LAST MODEL #
    ################### 
    
    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score, prototypes, count_features)
        save_ckpt(f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}_backup{opts.backup}.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score, prototypes, count_features)
        logger.info("[!] Checkpoint saved.")

    if not (opts.where_to_sim == 'GPU_windows' or opts.where_to_sim == 'CPU_windows'):
        torch.distributed.barrier()

    ###################
    # FINAL TEST      #
    ################### 
    # xxx From here starts the test code       
    
    logger.info("*** Test the model on all seen classes on "
                f"{opts.target_dataset if opts.uda_validate_target else opts.dataset}...")
    
    # load test model
    if TRAIN:
        model, model_params = train_helper.get_model(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step), opts)


        # Put the model on GPU
        if opts.where_to_sim == 'GPU_server':
            DistributedDataParallel(model.cuda(device))
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)

        ckpt = f"{logdir_full}/{task_name}-{opts.name}_{opts.step}{'_r'+str(opts.round) if opts.round > 0 else ''}_backup{opts.backup}.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")
        load_ckpt(checkpoint, model, strict=False)

        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = trainerClass(model, model_old, device=device, opts=opts, logdir=logdir_full)

    model.eval()

    # test 
    vis_dir = os.path.join(logdir_full,f"test-pics-step{opts.step}-backup{opts.backup}")
    os.makedirs(vis_dir, exist_ok=True)
    val_loss, val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger,
                                              world_size=world_size, vis_dir=vis_dir, label2color=label2color, denorm=denorm2, print_int=opts.val_print_interval)

    # print final parameters
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_table("Test_Class_Acc", val_score['Class Prc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    results["T-Prc"] = val_score['Class Prc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)        
    logger.add_scalar("T_MeanPrc", val_score['Mean Prc'], opts.step)        

    logger.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    if opts.multi:
        opts.name += '-multi'
    task_name = f"{opts.task}-{opts.dataset}"   
    
    logdir_full = f"{opts.logdir}/{task_name}/{task_name}-{opts.name}/"
    os.makedirs(f"{opts.logdir}/{task_name}", exist_ok=True)
    os.makedirs(f"{logdir_full}", exist_ok=True)

    main(opts)
    print('TOTAL TIME: ', time.time() - start_time)
