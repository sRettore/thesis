import torch
import torch.nn as nn

from losses.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss, FeaturesClusteringSeparationLoss, SNNL, \
    DistillationEncoderLoss, DistillationEncoderPrototypesLoss, FeaturesSparsificationLoss, BGRUncertaintyLoss, \
    KnowledgeDistillationCELossWithGradientScaling
from losses.loss_source import IWCrossEntropy, UnbiasedIWCrossEntropy, UnbiasedIWKnowledgeDistillationLoss
from utils.run_utils import *

class LossSourceHandler:
    def __init__(self, device, opts, has_model_old = False, num_classes=0, old_classes=0, logdir=None):

        self.step = opts.step
        self.no_mask = opts.no_mask  # if True sequential dataset from https://arxiv.org/abs/1907.13372
        self.overlap = opts.overlap
        self.loss_de_prototypes_sumafter = opts.loss_de_prototypes_sumafter
        self.num_classes = num_classes
        self.old_classes = old_classes
        
        self.loss_dict = {
            'loss': torch.tensor(0.),
            'lkd': torch.tensor(0.),
            'lde': torch.tensor(0.),
            'liCarl':  torch.tensor(0.),
            'lfc': torch.tensor(0.),
            'lsepClusters': torch.tensor(0.),
            'lSNNL': torch.tensor(0.),
            'ldeprototype': torch.tensor(0.),
            'lfs': torch.tensor(0.),
            'lbu': torch.tensor(0.),
            'lCIL': torch.tensor(0.)
            }

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        self.lIWce = (opts.src_lIWce > 0.)
        
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        else:
            if self.lIWce:
                if opts.unce and self.old_classes != 0:
                    self.criterion = UnbiasedIWCrossEntropy(ratio=opts.src_lIWce, old_cl=self.old_classes, ignore_index=255, reduction=reduction)
                else:
                    self.criterion = IWCrossEntropy(ratio=opts.src_lIWce, ignore_index=255, reduction=reduction)           
            else:
                if opts.unce and self.old_classes != 0:
                    self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
                else:
                    self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # features clustering loss
        self.lfc = opts.loss_fc
        self.lfc_flag = self.lfc > 0.
        # Separation between clustering loss
        self.lfc_sep_clust = opts.lfc_sep_clust
        self.lfc_loss = FeaturesClusteringSeparationLoss(num_classes= self.num_classes,
                                                         logdir=logdir if logdir is not None else '', feat_dim=2048,
                                                         device=device, lfc_L2normalized=opts.lfc_L2normalized,
                                                         lfc_nobgr=opts.lfc_nobgr, lfc_sep_clust=self.lfc_sep_clust,
                                                         lfc_sep_clust_ison_proto=opts.lfc_sep_clust_ison_proto,
                                                         orth_sep=opts.lfc_orth_sep, lfc_orth_maxonly=opts.lfc_orth_maxonly)

        # SNNL loss at features space
        self.lSNNL = opts.loss_SNNL
        self.lSNNL_flag = self.lSNNL > 0.
        if self.num_classes > 0 and logdir is not None:
            self.lSNNL_loss = SNNL(num_classes= self.num_classes, logdir=logdir, feat_dim=2048, device=device)

        # ILTSS paper loss: http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
        # https://arxiv.org/abs/1911.03462
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and has_model_old
        self.lde_loss = DistillationEncoderLoss(mask=opts.loss_de_maskedold, loss_de_cosine=opts.loss_de_cosine)

        self.ldeprototype = opts.loss_de_prototypes
        self.ldeprototype_flag = self.ldeprototype > 0.
        self.ldeprototype_loss = DistillationEncoderPrototypesLoss(num_classes=self.num_classes,
                                                                   device=device)

        # CIL paper loss: https://arxiv.org/abs/2005.06050
        self.lCIL = opts.loss_CIL
        self.lCIL_flag = self.lCIL > 0. and has_model_old
        self.lCIL_loss = KnowledgeDistillationCELossWithGradientScaling(temp=1, gs=self.lCIL, device=device, norm=False)

        # Features Sparsification Loss
        self.lfs = opts.loss_featspars
        self.lfs_flag = self.lfs > 0.
        self.lfs_loss = FeaturesSparsificationLoss(lfs_normalization=opts.lfs_normalization,
                                                   lfs_shrinkingfn=opts.lfs_shrinkingfn,
                                                   lfs_loss_fn_touse=opts.lfs_loss_fn_touse)

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and has_model_old
        self.lkd_features = self.lkd_flag and opts.lkd_features
        if opts.unkd:
            if opts.src_lIWkd:
                print("with ratio: ", opts.src_lIWkd)
                self.lkd_loss = UnbiasedIWKnowledgeDistillationLoss(alpha=opts.alpha, ratio = opts.src_lIWkd)
            else:
                self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)
                
        self.lbu = opts.loss_bgruncertainty
        self.lbu_flag = self.lbu > 0. and has_model_old
        self.lbu_loss = BGRUncertaintyLoss(device=device, num_classes=self.num_classes)
        self.lbu_inverse = opts.lbu_inverse
        self.lbu_mean = opts.lbu_mean

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and has_model_old
            self.icarl_only_dist = opts.icarl_disjoint and has_model_old
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined
        
        self.loss_enabled = {
            'loss': True,
            'lkd': self.lkd_flag,
            'lde': self.lde_flag,
            'liCarl':  self.icarl_combined,
            'lfc': self.lfc_flag,
            'lsepClusters': self.lfc_sep_clust,
            'lSNNL': self.lSNNL_flag,
            'ldeprototype': self.ldeprototype_flag,
            'lfs': self.lfs_flag,
            'lbu': self.lbu_flag,
            'lCIL': self.lCIL_flag
            }

    def require_intermediate(self):
        return self.lde or self.lfc or self.lfc_sep_clust or self.lSNNL or self.ldeprototype or \
            self.lfs or self.lbu or self.lCIL or self.lkd_features

    def require_model_old(self):
        return  self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.lfc_flag or \
            (self.lfc_sep_clust != 0.0) or self.lSNNL_flag or self.ldeprototype_flag or self.lbu_flag or self.lCIL

    def require_running_stats(self):
        return  self.lfc_flag or self.ldeprototype_flag or (self.lfc_sep_clust != 0.0)
    
    def sum_from_dict(self, to, keys):
        for k in keys:
            to += self.loss_dict[k].item()
        return to
    
    def reset_dict(self):
        self.loss_dict['loss'] = torch.tensor(0.)
        self.loss_dict['lkd'] = torch.tensor(0.)
        self.loss_dict['lde'] = torch.tensor(0.)
        self.loss_dict['liCarl'] = torch.tensor(0.)
        self.loss_dict['lfc'] = torch.tensor(0.)
        self.loss_dict['lsepClusters'] = torch.tensor(0.)
        self.loss_dict['lSNNL'] = torch.tensor(0.)
        self.loss_dict['ldeprototype'] = torch.tensor(0.)
        self.loss_dict['lfs'] = torch.tensor(0.)
        self.loss_dict['lbu'] = torch.tensor(0.)
        self.loss_dict['lCIL'] = torch.tensor(0.)
        
    def output_values(self, epoch_loss, reg_loss, interval_loss):
        epoch_loss += self.loss_dict['loss'].item()
        reg_loss = self.sum_from_dict(reg_loss, ['lkd', 'lde', 'liCarl','lfc', 'lSNNL','lsepClusters',
                                                 'ldeprototype', 'lfs', 'lbu', 'lCIL'])
        interval_loss = self.sum_from_dict(interval_loss, ['loss', 'lkd', 'lde', 'liCarl','lfc', 'lSNNL','lsepClusters',
                                                          'ldeprototype', 'lfs', 'lbu', 'lCIL'])
        return epoch_loss, reg_loss, interval_loss
    
    def lossText(self, prefix=''):
        lossText = f'{prefix}CE {self.loss_dict["loss"].item():.9f}'
        for key, value in self.loss_dict.items():
            if key != 'loss' and self.loss_enabled[key]:
                lossText += f',   {prefix}{key.upper()} {value.item():.9f}'
        return lossText
    
    def addToLogger(self, logger, x, prefix=''):
         for key, value in self.loss_dict.items():
            if key != 'loss' and self.loss_enabled[key]:
                logger.add_scalar(f'Losses/{prefix}{key}', value.item(), x)        

    def train_batch(self, cur_epoch, cur_step, labels, outputs, features = None, outputs_old = None, features_old = None, 
                    prototypes=None, count_features=None):        
        # xxx BCE / Cross Entropy Loss
        if not self.icarl_only_dist:
            self.loss_dict['loss'] = self.criterion(outputs, labels)  # B x H x W
        else:
            self.loss_dict['loss'] = self.licarl(outputs, labels, torch.sigmoid(outputs_old))
        self.loss_dict['loss'] = self.loss_dict['loss'].mean()  # scalar

        if self.icarl_combined:
            # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
            n_cl_old = outputs_old.shape[1]
            # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
            self.loss_dict['liCarl'] = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old))

        # features clustering loss
        if self.lfc_flag or self.lfc_sep_clust:
            self.loss_dict['lfc'], self.loss_dict['lsepClusters'] = self.lfc_loss(labels=labels, outputs=outputs,
                                                                                  features=features['body'], train_step=cur_step, 
                                                                                  step=self.step, epoch=cur_epoch, 
                                                                                  incremental_step=self.step, prototypes=prototypes)
        
        self.loss_dict['lfc'] *= self.lfc
        if torch.isnan(self.loss_dict['lfc']):  self.loss_dict['lfc'] = torch.tensor(0.)
        self.loss_dict['lsepClusters'] *= self.lfc_sep_clust

        # SNNL loss at features space
        if self.lSNNL_flag:
            self.loss_dict['lSNNL'] = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                                   features=features['body'], train_step=cur_step,
                                                                   epoch=cur_epoch)

        # xxx ILTSS (distillation on features or logits)
        if self.lde_flag:
            self.loss_dict['lde'] = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                           labels=labels, classes_old=self.old_classes)

        if self.lCIL_flag:
            outputs_old_temp = torch.zeros_like(outputs)
            outputs_old_temp[:,:outputs_old.shape[1],:,:] = outputs_old

            self.loss_dict['lCIL'] = self.lCIL_loss(outputs=outputs, targets=outputs_old_temp, targets_new=labels)

        if self.ldeprototype_flag:
            self.loss_dict['ldeprototype'] = self.ldeprototype * self.ldeprototype_loss(features=features['body'],
                                                                                        features_old=features_old[
                                                                                            'body'] if self.step != 0 else None,
                                                                                        labels=labels,
                                                                                        classes_old=self.old_classes,
                                                                                        incremental_step=self.step,
                                                                                        sequential=self.no_mask,
                                                                                        overlapped=self.overlap,
                                                                                        outputs_old=outputs_old if self.step != 0 else None,
                                                                                        outputs=outputs,
                                                                                        loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                                        prototypes=prototypes,
                                                                                        count_features=count_features)

        # Features Sparsification Loss
        if self.lfs_flag:
            self.loss_dict['lfs'] = self.lfs * self.lfs_loss(features=features['body'], labels=labels)

        if self.lbu_flag:
            self.loss_dict['lbu'] = self.lbu * self.lbu_loss(outputs, outputs_old=outputs_old if self.step != 0 else None, 
                                                              labels=labels, classes_old=self.old_classes, incremental_step=self.step, 
                                                              lbu_inverse=self.lbu_inverse, lbu_mean=self.lbu_mean)

        if self.lkd_flag:
            if not self.lkd_features:
                self.loss_dict['lkd'] = self.lkd * self.lkd_loss(outputs, outputs_old)
            else:
                self.loss_dict['lkd'] = self.lkd * self.lkd_loss(features["body"], features_old["body"])        
                
        # xxx first backprop of previous loss (compute the gradients for regularization methods)
        loss_tot = self.loss_dict['loss'] + self.loss_dict['lkd'] + self.loss_dict['lde'] + \
                   self.loss_dict['liCarl'] + self.loss_dict['lfc'] + \
                   self.loss_dict['lsepClusters'] + self.loss_dict['lSNNL'] + \
                   self.loss_dict['ldeprototype'] + self.loss_dict['lfs'] + \
                   self.loss_dict['lbu'] + self.loss_dict['lCIL']
        
        return loss_tot

    def validate_batch(self, cur_step, labels, outputs, features = None, outputs_old = None, features_old = None,
                       prototypes=None, count_features=None):
        '''Do validation'''
        # xxx BCE / Cross Entropy Loss
        if not self.icarl_only_dist:
            self.loss_dict['loss'] = self.criterion(outputs, labels)  # B x H x W
        else:
            self.loss_dict['loss'] = self.licarl(outputs, labels, torch.sigmoid(outputs_old))
        self.loss_dict['loss'] = self.loss_dict['loss'].mean()  # scalar
        
        if self.icarl_combined:
            # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
            n_cl_old = outputs_old.shape[1]
            # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
            self.loss_dict['liCarl'] = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                          torch.sigmoid(outputs_old))

        # features clustering loss
        if self.lfc_flag or self.lfc_sep_clust:
            self.loss_dict['lfc'], self.loss_dict['lsepClusters'] = self.lfc_loss(labels=labels, outputs=outputs,
                                                                                  features=features['body'], val=True)
        self.loss_dict['lfc'] *= self.lfc
        if torch.isnan(self.loss_dict['lfc']):  self.loss_dict['lfc'] = torch.tensor(0.)
        self.loss_dict['lsepClusters'] *= self.lfc_sep_clust
        
        # SNNL loss at features space
        if self.lSNNL_flag:
            self.loss_dict['lSNNL'] = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                 features=features['body'], val=True)

        # xxx ILTSS (distillation on features or logits)
        if self.lde_flag:
            self.loss_dict['lde'] = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                                             labels=labels, classes_old=self.old_classes)

        # Features Sparsification Loss
        if self.lfs_flag:
            self.loss_dict['lfs'] = self.lfs * self.lfs_loss(features=features['body'], labels=labels, val=True)

        if self.lbu_flag:
            self.loss_dict['lbu'] = self.lbu * self.lbu_loss(outputs, outputs_old=outputs_old if self.step != 0 else None,
                                               labels=labels,
                                               classes_old=self.old_classes, incremental_step=self.step,
                                               lbu_inverse=self.lbu_inverse, lbu_mean=self.lbu_mean)
            
        if self.ldeprototype_flag:
            self.loss_dict['ldeprototype'] = self.ldeprototype * self.ldeprototype_loss(features=features['body'],
                                                                                        features_old=features_old[
                                                                                            'body'] if self.step != 0 else None,
                                                                                        labels=labels,
                                                                                        classes_old=self.old_classes,
                                                                                        incremental_step=self.step,
                                                                                        sequential=self.no_mask,
                                                                                        overlapped=self.overlap,
                                                                                        outputs_old=outputs_old if self.step != 0 else None,
                                                                                        outputs=outputs,
                                                                                        loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                                        val=True,
                                                                                        prototypes=prototypes,
                                                                                        count_features=count_features)
        if self.lkd_flag:
            if not self.lkd_features:
                self.loss_dict['lkd'] = self.lkd * self.lkd_loss(outputs, outputs_old)
            else:
                self.loss_dict['lkd'] = self.lkd * self.lkd_loss(features["body"], features_old["body"]) 

        if self.lCIL_flag:
            self.loss_dict['lCIL']  = self.lCIL_loss(outputs=outputs, targets=outputs_old, targets_new=labels)

