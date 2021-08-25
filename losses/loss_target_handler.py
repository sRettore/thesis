import torch

from losses.loss_target import CrossEntropy, SoftCrossEntropy, IWSoftCrossEntropy, MaxSquareloss, IWMaxSquareloss, IW_MaxSquareloss, GroupMaxSquareLoss, IWNewMaxSquareloss
from losses.loss import KnowledgeDistillationLoss, UnbiasedKnowledgeDistillationLoss
from utils.run_utils import *

class LossTargetHandler:
    def __init__(self, args, applySoftMax=True, num_classes=0, old_classes=0, val=0):
        self.num_classes = num_classes
        self.old_classes = old_classes
        
        # loss dictionary
        self.loss_dict = {
            'lce':  torch.tensor(0.),
            'lsce': torch.tensor(0.),
            'lIWsce': torch.tensor(0.),
            'lmsq': torch.tensor(0.),
            'lIWmsq':  torch.tensor(0.),
            'lkd':torch.tensor(0.),
            'lfmsq': torch.tensor(0.),
            'lfIWmsq':  torch.tensor(0.),
            }
                
        # loss activation flags
        lce_flag = (args['lce'] > 0.)
        lsce_flag = (args['lsce'] > 0.)
        lmsq_flag = (args['lmsq'] > 0.)
        lIWsce_flag = (args['lIWsce'] > 0.)
        lIWmsq_flag = (args['lIWmsq'] > 0.)
        lkd_flag = (args['lkd'] > 0.)
        lfmsq_flag = (args['lfmsq'] > 0.)
        lfIWmsq_flag = (args['lfIWmsq'] > 0.)
        
        self.lce_lambda = args['lce']
        self.lsce_lambda = args['lsce']
        self.lmsq_lambda = args['lmsq']  
        self.lIWsce_ratio = args['lIWsce']
        self.lIWmsq_ratio = args['lIWmsq']
        self.lkd_lambda = args['lkd']
        self.lkd_alpha = args['lkd_alpha'] if args['lkd_alpha'] > 0 else 1.
        self.unkd = args['unkd']
        self.group_msq = args['group_msq']
        self.new_msq = args['new_msq']
        self.lkd_features = args['lkd_features_target']
        self.lfmsq_lambda = args['lfmsq']
        self.lfIWmsq_ratio = args ['lfIWmsq']
        
        # loss flag dictionary
        self.loss_enabled = {
            'lce': lce_flag,
            'lsce': lsce_flag and not lIWsce_flag,
            'lIWsce': lsce_flag and lIWsce_flag,
            'lmsq': lmsq_flag and not lIWmsq_flag,
            'lIWmsq': lmsq_flag and lIWmsq_flag,
            'lfmsq': lfmsq_flag and not lfIWmsq_flag,
            'lfIWmsq': lfmsq_flag and lfIWmsq_flag,
            'lkd': lkd_flag
            }
        
        # loss definition
        if self.loss_enabled['lce']:
            self.lce = CrossEntropy(reduction='mean', ignore_index=255, applySoftMax=applySoftMax)
                
        if self.loss_enabled["lsce"]:
            self.lsce = SoftCrossEntropy(reduction = "mean", applySoftMax=applySoftMax)
        
        if self.loss_enabled['lIWsce']:
            self.lsce = IWSoftCrossEntropy(ratio=self.lIWsce_ratio, reduction = "mean", applySoftMax=applySoftMax)

        if self.loss_enabled["lmsq"]:
            self.lmsq = MaxSquareloss(reduction = "mean", applySoftMax=applySoftMax)
        
        if self.loss_enabled['lIWmsq']:
            if self.old_classes > 0:
                if val == 0:
                    val = self.old_classes
                
                if self.group_msq:
                    self.lmsq = GroupMaxSquareLoss(old_cl= val, ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
                    #self.lmsq = GroupMaxSquareLoss(old_cl= self.old_classes, ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
                elif self.new_msq:
                    print ("oldcl ", val)
                    self.lmsq = IWNewMaxSquareloss(old_cl= val, ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
                    #self.lmsq = IWNewMaxSquareloss(old_cl= self.old_classes, ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
                else:
                    self.lmsq = IWMaxSquareloss(ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
            else:
                self.lmsq = IWMaxSquareloss(ratio=self.lIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)

        if self.loss_enabled["lfmsq"]:
            self.lfmsq = MaxSquareloss(reduction = "mean", applySoftMax=applySoftMax)
        
        if self.loss_enabled['lfIWmsq']:
            self.lfmsq = IWMaxSquareloss(ratio=self.lfIWmsq_ratio, reduction = "mean", applySoftMax=applySoftMax)
                
        if self.loss_enabled['lkd']:
            if self.unkd:
                self.lkd = UnbiasedKnowledgeDistillationLoss(alpha=self.lkd_alpha)
            else:
                self.lkd = KnowledgeDistillationLoss(alpha=self.lkd_alpha)
    
    def sum_from_dict(self, to, keys):
        for k in keys:
            to += self.loss_dict[k].item()
        return to
    
    def reset_dict(self):
        self.loss_dict['lce'] = torch.tensor(0.)
        self.loss_dict['lsce'] = torch.tensor(0.)
        self.loss_dict['lIWsce'] = torch.tensor(0.)
        self.loss_dict['lmsq'] = torch.tensor(0.)
        self.loss_dict['lIWmsq'] = torch.tensor(0.)
        self.loss_dict['lkd'] = torch.tensor(0.)
        
    def output_values(self, epoch_loss, reg_loss, interval_loss):
        reg_loss = self.sum_from_dict(reg_loss, ['lce', 'lsce', 'lIWsce', 'lmsq', 'lIWmsq', 'lkd', 'lfmsq', 'lfIWmsq'])
        interval_loss = self.sum_from_dict(interval_loss, ['lce', 'lsce', 'lIWsce', 'lmsq', 'lIWmsq', 'lkd', 'lfmsq', 'lfIWmsq'])
        return epoch_loss, reg_loss, interval_loss
    
    def lossText(self, prefix=''):
        lossText = ''
        for key, value in self.loss_dict.items():
            if self.loss_enabled[key]:
                lossText += f',   {prefix}{key.upper()} {value.item():.9f}'
        return lossText
    
    def addToLogger(self, logger, x, prefix=''):
         for key, value in self.loss_dict.items():
            if self.loss_enabled[key]:
                logger.add_scalar(f'Losses/{prefix}{key}', value.item(), x)        

    def require_model_old(self):
        return self.loss_enabled['lkd']
        
    def require_intermediate(self):
        return self.lkd_features or self.loss_enabled['lfmsq'] or self.loss_enabled['lfIWmsq']
   
    def train_batch(self, cur_epoch, cur_step, outputs, outputs_old=None, features = None, features_old = None, prediction=None):       
        if self.loss_enabled['lce']:
            self.loss_dict['lce'] =  self.lce_lambda * self.lce(inputs=outputs, targets=prediction)  
        
        if self.loss_enabled['lsce']:
            # entropy loss with targets the probabilities themselves
            self.loss_dict['lsce'] =  self.lsce_lambda * self.lsce(inputs=outputs, targets=outputs) 
            
        if self.loss_enabled['lIWsce']:
            # entropy loss with targets the probabilities themselves
            self.loss_dict['lIWsce'] =  self.lsce_lambda * self.lsce(inputs=outputs, targets=outputs)
            
        if self.loss_enabled['lmsq']:
            self.loss_dict['lmsq'] = self.lmsq_lambda * self.lmsq(inputs=outputs)
          
        if self.loss_enabled['lIWmsq']:
            self.loss_dict['lIWmsq'] = self.lmsq_lambda * self.lmsq(inputs=outputs)

        if self.loss_enabled['lfmsq']:
            self.loss_dict['lfmsq'] = self.lfmsq_lambda * self.lfmsq(inputs=features["body"])
           
        if self.loss_enabled['lfIWmsq']:
            self.loss_dict['lfIWmsq'] = self.lfmsq_lambda * self.lfmsq(inputs=features["body"])
            
        if self.loss_enabled['lkd']:
            if not self.lkd_features:
                self.loss_dict['lkd'] = self.lkd_lambda * self.lkd(inputs=outputs, targets=outputs_old)
            else:
                self.loss_dict['lkd'] = self.lkd_lambda * self.lkd(inputs = features["body"], targets = features_old["body"])              

        loss_tot = self.loss_dict['lce'] + \
                   self.loss_dict['lsce'] + self.loss_dict['lIWsce'] + \
                   self.loss_dict['lmsq'] + self.loss_dict['lIWmsq'] + \
                   self.loss_dict['lkd'] + \
                   self.loss_dict['lfmsq'] + self.loss_dict['lfIWmsq']
        return loss_tot

    def validate_batch(self, cur_step, outputs, outputs_old=None, features = None, features_old = None, prediction=None):      
        if self.loss_enabled['lce']:
            self.loss_dict['lce'] =  self.lce_lambda * self.lce(inputs=outputs, targets=prediction)              
        
        if self.loss_enabled['lsce']:
            # entropy loss with targets the probabilities themselves
            self.loss_dict['lsce'] =  self.lsce_lambda * self.lsce(inputs=outputs, targets=outputs) 
            
        if self.loss_enabled['lIWsce']:
            # entropy loss with targets the probabilities themselves
            self.loss_dict['lIWsce'] =  self.lsce_lambda * self.lsce(inputs=outputs, targets=outputs)
            
        if self.loss_enabled['lmsq']:
            self.loss_dict['lmsq'] = self.lmsq_lambda * self.lmsq(inputs=outputs)
            
        if self.loss_enabled['lIWmsq']:
            self.loss_dict['lIWmsq'] = self.lmsq_lambda * self.lmsq(inputs=outputs)

        if self.loss_enabled['lfmsq']:
            if features is not None and features_old is not None:
                self.loss_dict['lfmsq'] = self.lfmsq_lambda * self.lfmsq(inputs=features["body"])
           
        if self.loss_enabled['lfIWmsq']:
            if features is not None and features_old is not None:
                self.loss_dict['lfIWmsq'] = self.lfmsq_lambda * self.lfmsq(inputs=features["body"])
            
        if self.loss_enabled['lkd']:
            if not self.lkd_features:
                self.loss_dict['lkd'] = self.lkd_lambda * self.lkd(inputs=outputs, targets=outputs_old)
            else:
                if features is not None and features_old is not None:
                    self.loss_dict['lkd'] = self.lkd_lambda * self.lkd(inputs = features["body"], targets = features_old["body"])             
