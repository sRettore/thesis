import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).sum(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
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


# CIL loss defined in paper: https://arxiv.org/abs/2005.06050
class KnowledgeDistillationCELossWithGradientScaling(nn.Module):
    def __init__(self, temp=1, device=None, gs=1, norm=False):
        """Initialises the loss

                :param temp: temperature of the knowledge distillation loss, reduces to CE-loss for t = 1
                :param device: torch device used during training
                :param gs: defines the strength of the scaling
                :param norm: defines how the loss is normalized

        """

        super().__init__()
        assert isinstance(temp, int), "temp has to be of type int, default is 1"
        assert isinstance(device, torch.device), "device has to be of type torch.device"
        # assert gs > 0, "gs has to be > 0"
        assert isinstance(norm, bool), "norm has to be of type bool"

        self.temp = temp
        self.device = device
        self.gs = gs
        self.norm = norm

    def forward(self, outputs, targets, targets_new=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"

        """forward function

                        output: output of the network
                        targets: soft targets from the teacher
                        targets_new: hard targets for the new classes

                """
        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...

        # here the weights are calculated as described in the paper, just remove the weights from the calculation as
        # in KnowledgeDistillationCELoss
        targets = torch.softmax(targets, dim=1)
        outputs = torch.softmax(outputs, dim=1)
        denom_corr = 0
        ln2 = torch.log(torch.tensor([2.0]).to(self.device))  # basis change
        entropy = -torch.sum(targets * torch.log(targets+1e-8), dim=1, keepdim=True) / ln2
        weights = entropy * self.gs + 1

        # calculate the mask from the new targets, so that only the regions without labels are considered
        if targets_new is not None:
            mask = torch.zeros(targets_new.shape).to(self.device)
            mask[targets_new == 255] = 1
            mask[targets_new == 0] = 1
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            weights = mask * weights
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = weights * torch.sum(targets * outputs, dim=1, keepdim=True)  # TODO: NB here there was a minus sign

        # Apply mean reduction
        if self.norm:
            denom = torch.sum(weights)
        else:
            denom = torch.numel(loss[:, 0, ...]) - denom_corr
        loss = torch.sum(loss) / (denom+1e-8)

        return self.temp**2 * loss  # Gradients are scaled down by 1 / T if not corrected




class SNNL(nn.Module):
    def __init__(self, device, reduction='mean', alpha=1., num_classes=0, logdir=None, feat_dim=2048):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        self.logdir = logdir

    def forward(self, labels, outputs, features, train_step, epoch, val=False, mask=None): #features_old, mask=None):
        loss = torch.tensor(0., device=self.device)
        temperature = 1
        # outputs = torch.softmax(outputs, dim=1)

        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()

        x = features.view(-1, features.shape[1])  # bF*2048
        y = labels_down.view(-1)  # bF

        cl_present = torch.unique(input=labels_down)

        r = 0
        for i in range(x.shape[0]):
            numerator = denominator = 0
            xi = x[i,:]
            for j in range(x.shape[0]):
                xj = x[j,:]
                if j != i:
                    if y[i] == y[j]:
                        numerator += torch.exp(-torch.norm(xi-xj)**2 / temperature)

                    denominator += torch.exp(-torch.norm(xi-xj)**2 / temperature)

            r += (torch.log(numerator/denominator))

        loss = - r/x.shape[0]

        return loss


class FeaturesClusteringSeparationLoss(nn.Module):
    def __init__(self, device, reduction='mean', alpha=1., num_classes=0, logdir=None, feat_dim=2048,
                 lfc_L2normalized=False, lfc_nobgr=False, lfc_sep_clust=0., lfc_sep_clust_ison_proto=False,
                 orth_sep=False, lfc_orth_maxonly=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        self.logdir = logdir
        self.lfc_L2normalized = lfc_L2normalized
        self.lfc_nobgr = lfc_nobgr
        self.lfc_sep_clust = lfc_sep_clust
        self.lfc_sep_clust_ison_proto = lfc_sep_clust_ison_proto
        self.orth_sep = orth_sep
        self.lfc_orth_maxonly = lfc_orth_maxonly

    def forward(self, labels, outputs, features, train_step=0, step=0, epoch=0, val=False, mask=None, prototypes=None,
                incremental_step=None):
        loss_features_clustering = torch.tensor(0., device=self.device)
        loss_separationclustering = torch.tensor(0., device=self.device)

        return loss_features_clustering, loss_separationclustering


class DistillationEncoderLoss(nn.Module):
    def __init__(self, mask=False, loss_de_cosine=False):
        super().__init__()
        self.mask = mask
        self.loss_de_cosine = loss_de_cosine

    def forward(self, features, features_old, labels, classes_old):
        if not self.loss_de_cosine:
            loss_to_use = nn.MSELoss(reduction='none')
            loss = loss_to_use(features, features_old)
            if self.mask:
                masked_features = self._compute_mask_old_classes(features, labels, classes_old)
                loss = loss[masked_features.expand_as(loss)]
        else:
            loss_to_use = nn.CosineSimilarity()
            loss = 1.0 - loss_to_use(features, features_old)
            if self.mask:
                masked_features = self._compute_mask_old_classes(features, labels, classes_old)
                loss = loss[masked_features.squeeze()]

        outputs = torch.mean(loss)

        return outputs


class DistillationEncoderPrototypesLoss(nn.Module):  # IDEA 1b
    def __init__(self, device, num_classes, mask=False):
        super().__init__()
        self.mask = mask
        self.num_classes = num_classes
        self.device = device


    def forward(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None):
        outputs = torch.tensor(0., device=self.device)

        return outputs


class FeaturesSparsificationLoss(nn.Module):  # IDEA 1b
    def __init__(self, lfs_normalization, lfs_shrinkingfn, lfs_loss_fn_touse, mask=False, reduction='mean'):
        super().__init__()
        self.mask = mask
        self.lfs_normalization = lfs_normalization
        self.lfs_shrinkingfn = lfs_shrinkingfn
        self.lfs_loss_fn_touse = lfs_loss_fn_touse
        self.eps = 1e-15
        self.reduction = reduction

    def forward(self, features, labels):
        outputs = torch.tensor(0.)

        return outputs



class BGRUncertaintyLoss(nn.Module):
    # Background uncertainty loss. If previous model is uncertain on the estimation of the background in region
    # where there is a new class then it is ok. Otherwise penalize it.
    def __init__(self, device, num_classes, mask=False):
        super().__init__()
        self.mask = mask
        self.num_classes = num_classes
        self.device = device


    def forward(self, outputs, outputs_old, labels, classes_old, incremental_step, lbu_inverse=False, lbu_mean=False):
        loss = torch.tensor(0., device=self.device)


        return loss
