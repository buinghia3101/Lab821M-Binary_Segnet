import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_factors=(1, 2, 3, 6), batch_norm=True):
        super().__init__()
        self.spatial_blocks = nn.ModuleList([
            self._make_spatial_block(in_channels, pf, batch_norm) for pf in pool_factors
        ])
        layers = [nn.Conv2d(in_channels * (len(pool_factors) + 1), out_channels, kernel_size=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.Tanh()]
        self.bottleneck = nn.Sequential(*layers)

    def _make_spatial_block(self, in_channels, pool_factor, batch_norm):
        layers = [
            nn.AdaptiveAvgPool2d((pool_factor, pool_factor)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(in_channels)]
        layers += [nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_outs = [x]
        for block in self.spatial_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = layer.weight.data.sign()
            pooled = block(x)
            pool_outs.append(F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False))
        o = torch.cat(pool_outs, dim=1)

        for layer in self.bottleneck:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = layer.weight.data.sign()

        return self.bottleneck(o)


class PSPUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.Tanh()]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        for layer in self.layer:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = layer.weight.data.sign()

        return self.layer(x)


class PSPnet(nn.Module):
    def __init__(self, n_classes, pretrained_model: nn.Module, batch_norm=True, psp_out_feature=1024):
        super(PSPnet, self).__init__()
        self.features = pretrained_model.features

        for m in reversed(list(self.features.modules())):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
                break

        self.PSP = PSPModule(channels, out_channels=psp_out_feature, batch_norm=batch_norm)
        h = psp_out_feature // 2
        q = psp_out_feature // 4
        e = psp_out_feature // 8

        self.upsampling1 = PSPUpsampling(psp_out_feature, h, batch_norm=batch_norm)
        self.upsampling2 = PSPUpsampling(h, q, batch_norm=batch_norm)
        self.upsampling3 = PSPUpsampling(q, e, batch_norm=batch_norm)

        self.classifier = nn.Conv2d(e, n_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        for f in self.features:
            o = f(o)

        o = self.PSP(o)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)

        o = F.interpolate(o, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return self.classifier(o)
        return o


from ..encoders.vgg import *
from ..encoders.resnet import *
from ..encoders.mobilenet import *

def pspnet_vgg11(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_11(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg13(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_13(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg16(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_16(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg19(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_19(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)

def pspnet_resnet18(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet18(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet34(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet34(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet50(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet50(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet101(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet101(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet152(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet152(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)

def pspnet_mobilenet_v2(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    mobile_net = mobilenet(pretrained, fixed_feature)
    copy_feature_info = mobile_net.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    mobile_net.features = mobile_net.features[:squeeze_feature_idx]
    return PSPnet(n_classes, mobile_net, batch_norm)

