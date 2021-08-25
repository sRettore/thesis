# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, classes):
        super(Classifier_Module, self).__init__()
        self.classes = classes
        self.aspp_list = nn.ModuleList()
        self.clsf_list = nn.ModuleList()
        
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"    

        self.aspp_list.append(self._make_aspp_layer(inplanes, 256, dilation_series, padding_series))            
        for i in range(len(classes)):
            self.clsf_list.append(self._make_cls_layer(256, classes[i]))
    
    def _make_aspp_layer(self, inplanes, outplanes, dilation_series, padding_series):
        conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            conv2d_list.append(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        
        for m in conv2d_list:
            m.weight.data.normal_(0, 0.01) 
        return conv2d_list
    
    def _make_cls_layer(self, inplanes, classes):
        cls = nn.ModuleList(
            [nn.Conv2d(inplanes, classes, 1)]
        )       
        for m in cls:
            m.weight.data.normal_(0, 0.01)
        return cls
                
    def forward(self, x):       
        outcls = []
        prelogits = []
        for i in range(len(self.clsf_list)):
            conv2d_list = self.aspp_list[0]
            cls = self.clsf_list[i][0]
            
            out = conv2d_list[0](x)
            for j in range(len(conv2d_list) - 1):
                out += conv2d_list[j + 1](x)
            
            prelogits.append(out)
            out = cls(out)
            outcls.append(out)
        out_logits = torch.cat(outcls, dim=1)
        pre_logits = torch.cat(prelogits, dim=1)
        return out_logits, pre_logits

    def init_new_classifier(self, device):
        cls = self.clsf_list[-1]
        aspp = self.aspp_list[-1]
        aspp0 = self.aspp_list[0]
        cls0 = self.clsf_list[0]
        
        #for j in range(len(aspp)):
        #    self.init_new_aspp(device, aspp[j], aspp0[j], classifier_my)
        
        self.init_new_clsf(device, cls[0], cls0[0])              
        
    def init_new_aspp(self, device, convNew, convOld):
        imprinting_w = convOld.weight.data.clone()
        bkg_bias = convOld.bias.data.clone()
        
        convNew.weight.data.copy_(imprinting_w)
        convNew.bias.data.copy_(bkg_bias)

    def init_new_clsf(self, device, convNew, convOld):
        imprinting_w = convOld.weight.data[0].clone()
        bkg_bias = convOld.bias.data[0].clone()
        
        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias-bias_diff)

        convNew.weight.data.copy_(imprinting_w)
        convNew.bias.data.copy_(new_bias)
        convOld.bias.data.copy_((convOld.bias.data-bias_diff).squeeze(0))              
            
class ResNetMulti(nn.Module):
    def __init__(self, block, layers, classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, classes):
        return block(inplanes, dilation_series, padding_series, classes)

    def forward(self, x, ret_intermediate=False):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_body_low = self.layer3(x)
        x1, x_prelogits_low = self.layer5(x_body_low)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        x_body_high = self.layer4(x_body_low)
        x2, x_prelogits_high = self.layer6(x_body_high)
        x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)

        if ret_intermediate:
            return x2, x1, {"body": x_body_high, "pre_logits": x_prelogits_high}, {"body": x_body_low, "pre_logits": x_prelogits_low}, 
            
        return x2, x1 # changed!

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():                   
                    if k.requires_grad:
                        yield k
                        
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
        
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
               {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]
        
    def init_new_classifier(self, device):
        self.layer5.init_new_classifier(device)
        self.layer6.init_new_classifier(device)

def DeeplabMulti(classes, pretrained=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], classes)

    if pretrained:
        restore_from = './pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth'
        saved_state_dict = torch.load(restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    return model