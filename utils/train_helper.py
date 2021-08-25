import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.deeplab_multi import DeeplabMulti

def get_model(classes, args):
    model = DeeplabMulti(classes=classes, pretrained=not args.no_pretrained)
    params = model.optim_parameters(args)
    return model, params