import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# cross entropy that can avoid using softmax but does log and that when there is no target automatically does
# hard cross entropy with predictions from the inputs 
class CrossEntropy(nn.Module):
    def __init__(self, reduction = "mean", ignore_index = -1, applySoftMax=True):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction
        self.applySoftMax = applySoftMax
        self.ignore_index = ignore_index
        
        if applySoftMax:
            self.net = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            self.net = nn.NLLLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(self, inputs, targets=None):
        """
        :param inputs: predictions (N, C, H, W)
        :param targets: target distribution labels
        :return: loss
        """
        if not self.applySoftMax:
            inputs = torch.log(inputs)
        
        # self hard cross entropy with predictions for (hard-)entropy minimization uda
        if targets is None:
            _, targets = inputs.max(dim=1)
        
        return self.net(input=inputs, target=targets)

class SoftCrossEntropy(nn.Module):
    def __init__(self, reduction = "mean", applySoftMax=True):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reduction
        self.applySoftMax = applySoftMax

    def forward(self, inputs, targets):
        """
        :param inputs: predictions (N, C, H, W)
        :param targets: target distribution (N, C, H, W)
        :return: loss
        """
            
        assert inputs.size() == targets.size()
        
        if self.applySoftMax:
            log_likelihood = F.log_softmax(inputs, dim=1)
            targets = F.softmax(targets, dim=1)
        else:
            log_likelihood = torch.log(inputs)
            
        loss = (-targets*log_likelihood).sum(dim=1)

        N, C, H, W = inputs.size()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class IWSoftCrossEntropy(nn.Module):
    # class_wise softCrossEntropy for class balance
    def __init__(self, ratio=0.2, reduction = "mean", applySoftMax = True):
        super().__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.applySoftMax = applySoftMax

    def forward(self, inputs, targets):
        
        assert inputs.size() == targets.size()

        N, C, H, W = inputs.size()

        if self.applySoftMax:
            log_likelihood = F.log_softmax(inputs, dim=1)
        else:
            log_likelihood = torch.log(inputs)

        _, argpred = torch.max(inputs, 1)
        weights = []
        for i in range(N):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            hist[hist == 0] = 1 # replace empty bins with 1 
            weight = (torch.pow( hist.sum()/hist, self.ratio)).to(argpred.device).detach()
            weights.append(weight)        
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)

        loss = (-targets*log_likelihood*weights).sum(dim=1)
			
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss()

class IWMaxSquareloss(nn.Module):
    def __init__(self, ratio=0.2, reduction="mean", applySoftMax = True):
        super().__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.applySoftMax = applySoftMax
    
    def forward(self, inputs, targets=None):
        """
        :param inputs: probability of pred (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        
        if self.applySoftMax:
            inputs = torch.softmax(inputs, dim=1)

        N, C, H, W = inputs.size()

        maxpred, argpred = torch.max(inputs, 1)
           
        weights = []
        for i in range(N):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            hist[hist == 0] = 1 # replace empty bins with 1 
            
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(argpred.device).detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)
        
        loss = (-torch.pow(inputs, 2)*weights) #/2
        #loss = loss.sum(dim=1)
                
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class IW_MaxSquareloss(nn.Module):
    def __init__(self, ratio=0.2, reduction="mean", applySoftMax = True):
        super().__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.applySoftMax = applySoftMax
        
    def forward(self, inputs, targets=None):
        """
        :param inputs: probability of pred (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        
        if self.applySoftMax:
            inputs = torch.softmax(inputs, dim=1)

        N, C, H, W = inputs.size()

        maxpred, argpred = torch.max(inputs, 1)
           
        weights = []
        for i in range(N):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        
        loss = (-torch.pow(inputs, 2)*weights)
                
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
class MaxSquareloss(nn.Module):
    def __init__(self, reduction='mean', applySoftMax = True):
        super().__init__()
        self.reduction = reduction
        self.applySoftMax = applySoftMax
        
    def forward(self, inputs, targets=None):
        """
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        
        if self.applySoftMax:
            inputs = torch.softmax(inputs, dim=1)
            
        N, C, H, W = inputs.size()
        loss = (-torch.pow(inputs, 2)) /2
        loss = loss.sum(dim=1)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class GroupMaxSquareLoss(nn.Module):
    def __init__(self, old_cl, ratio=0.2, reduction="mean", applySoftMax=True):
        super().__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.applySoftMax = applySoftMax
        self.old_cl = old_cl
        
    def forward(self, inputs, targets=None):
        """
        :param inputs: probability of pred (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        
        if self.applySoftMax:
            inputs = torch.softmax(inputs, dim=1)
            
        N, C, H, W = inputs.size()
         
        maxpred, argpred = torch.max(inputs, 1)

        #print('inputs ', inputs[0,:,0,0])
        if self.old_cl > 0:        
            inputs_group = torch.zeros((N,C,H,W), dtype=inputs.dtype, device = inputs.device)    
            inputs_group[:,0] = torch.sum(inputs[:,:self.old_cl], dim=1)
            inputs_group[:,self.old_cl:] = inputs[:,self.old_cl:]
            #print('group ',inputs_group[0,:,0,0])
        else:
            inputs_group = inputs
            
        weights = []
        for i in range(N):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            hist[0] = torch.sum(hist[:self.old_cl])
            hist[hist == 0] = 1 # replace empty bins with 1
            #print('hist', hist)
            
            if self.old_cl > 0:
                idx_old_cl = (hist < 0)
                idx_old_cl[1:self.old_cl] = True        
                hist[idx_old_cl] = 0
                
                weight = (torch.pow(hist.sum() / hist, self.ratio)).to(argpred.device).detach()
                weight[idx_old_cl] = 0
            else:
                weight = (torch.pow(hist.sum() / hist, self.ratio)).to(argpred.device).detach()  
                
            #print('weight ', weight)                
            weights.append(weight)
        weights_group = torch.stack(weights, dim=0).unsqueeze_(2).unsqueeze_(3)
        
        loss = -torch.pow((inputs_group), 2)*weights_group
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
class IWNewMaxSquareloss(nn.Module):
    def __init__(self, old_cl, ratio=0.2, reduction="mean", applySoftMax=True):
        super().__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.applySoftMax = applySoftMax
        self.old_cl = old_cl
        
    def forward(self, inputs, targets=None):
        """
        :param inputs: probability of pred (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        
        if self.applySoftMax:
            inputs = torch.softmax(inputs, dim=1)
            
        N, C, H, W = inputs.size()
         
        maxpred, argpred = torch.max(inputs, 1)
            
        weights = []
        for i in range(N):
            hist = torch.histc(argpred[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            hist[hist == 0] = 1 # replace empty bins with 1
            
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(argpred.device).detach()

            if self.old_cl > 0:
                weight[:self.old_cl] = 1
                
            #print('weight ', weight)                
            weights.append(weight)
        weights_group = torch.stack(weights, dim=0).unsqueeze_(2).unsqueeze_(3)
        
        loss = -torch.pow((inputs), 2)*weights_group
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss