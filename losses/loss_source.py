import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        old_cl = targets.shape[1]
        new_cl = inputs.shape[1] - targets.shape[1]
   
        targets = targets * self.alpha
   
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        
        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bkg = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bkg).sum(dim=1)) #/ targets.shape[1]

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
        
class UnbiasedIWKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., ratio = 0.1):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.ratio = ratio

    def forward(self, inputs, targets, mask=None):
        old_cl = targets.shape[1]
        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha
   
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        
        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bkg = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        B, C, H, W = inputs.size()
                   
        weights = []
        for i in range(B):
            hist = torch.histc(targets[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
                            
            hist[hist == 0] = 1 # replace empty bins with 1
            hist[0] = hist[0] + hist[old_cl:].sum()
            hist[old_cl:] = 0
                        
            weight = (torch.pow(hist / hist.sum(), self.ratio)).to(targets.device).detach()
            weights.append(weight[:old_cl])
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)

        
        labels = weights * torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W
        
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bkg).sum(dim=1)) #/ targets.shape[1]

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
        
class UnbiasedIWTrueKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., ratio = 0.1):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.ratio = ratio

    def forward(self, inputs, targets, mask=None):
        old_cl = targets.shape[1]
        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha
   
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        
        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bkg = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        B, C, H, W = inputs.size()
                   
        weights = []
        for i in range(B):
            hist = torch.histc(targets[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
                            
            #print("hist1: ", hist)
            hist[hist == 0] = 1 # replace empty bins with 1
            temp = hist[old_cl:].sum()
            hist[old_cl:] = 0
            hist[0] = hist[0] + temp
            
            #print("hist2: ", hist)
            
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(targets.device).detach()
            #print("weight: ", weight)
            weights.append(weight[:old_cl])
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)
        
        #print("weights: ", weights)
        #print("labels: ",torch.softmax(targets, dim=1)[0,:,0,0])
        
        labels = weights * torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W
        #print("labels2: ",labels[0,:,0,0])
        
        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bkg).sum(dim=1)) #/ targets.shape[1]

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
    
class IWCrossEntropy(nn.Module):
    def __init__(self, ratio = 1.0, reduction='mean', ignore_index=255, applySoftMax=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ratio = ratio
        self.applySoftMax = applySoftMax

    def forward(self, inputs, targets):

        if self.applySoftMax:
            outputs = F.log_softmax(inputs, dim=1)
        else:
            outputs = F.log(inputs)
        
        N, C, H, W = outputs.size()
                   
        weights = []
        for i in range(N):
            hist = torch.histc(targets[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            hist[hist == 0] = 1 # replace empty bins with 1
            
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(targets.device).detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)
        
        outputs = (outputs*weights)
        
        loss = F.nll_loss(outputs, targets, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss

class UnbiasedIWCrossEntropy(nn.Module):
    def __init__(self, ratio = 1.0, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl
        self.ratio = ratio
        self.norm = norm

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero
        #print("labels:",labels)
        
        N, C, H, W = outputs.size()
                   
        weights = []
        for i in range(N):
            hist = torch.histc(labels[i].cpu().data.float(), 
                            bins=C, min=0,
                            max=C-1).float()
            idx0 = (hist == 0)
            idx_old_cl = (hist < 0)
            idx_old_cl[:old_cl] = True
            idx_old_cl[0]=False
            hist[0] = hist[:old_cl].sum()
            hist[idx_old_cl]=0
            hist[idx0 & ~idx_old_cl] = 1 # replace empty bins with 1
            
            weight = (torch.pow(hist.sum() / hist, self.ratio)).to(labels.device).detach()
            weight[idx_old_cl]=0
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze_(2).unsqueeze_(3)
        
        outputs = (outputs*weights)        
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss