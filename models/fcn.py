from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.segmentation import model_urls, load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter


class FCNHead(nn.Sequential):
    def __init__(self, img_shape, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)
        self.img_shape = img_shape

    def forward(self, inputs):
        output = super(FCNHead, self).forward(inputs)
        output = F.interpolate(output, size=self.img_shape, mode='bilinear', align_corners=False)
        return output


class FCN(nn.Module):
    def __init__(self, img_shape, backbone, classifier):
        super(FCN, self).__init__()
        self.img_shape = img_shape
        self.encoder = backbone
        self.g = classifier

    def forward(self, x):
        features = self.encoder(x)
        output = self.g(features)
        return output


class MyIntermediateLayerGetter(IntermediateLayerGetter):
    def __init__(self, model, return_layers: str):
        super(MyIntermediateLayerGetter, self).__init__(model, {return_layers: 'out'})
        self.return_layers = return_layers

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
            if name == self.return_layers:
                out = x
        return out


def _segm_resnet(backbone_name, img_shape, num_classes, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = 'layer4'
    backbone = MyIntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = FCNHead(img_shape, inplanes, num_classes)

    model = FCN(img_shape, backbone, classifier)
    return model


def make_fcn(pretrained=False, progress=True, img_shape=(512, 1024), num_classes=21, **kwargs):
    model = _segm_resnet('resnet50', img_shape, num_classes, **kwargs)
    if pretrained:
        arch = 'fcn_resnet50_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            del state_dict['classifier.4.weight']
            del state_dict['classifier.4.bias']
            model.load_state_dict(state_dict, strict=False)
    return model
