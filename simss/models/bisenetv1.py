import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones import BACKBONES
from .losses import OHEMCELoss


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 normalize: bool = True, activate: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not normalize)
        if normalize:
            self.bn = nn.BatchNorm2d(out_channels)
        if activate:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'act'):
            out = self.act(x)
        else:
            out = x
        return out


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvModule(channels, channels, kernel_size=1, activate=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_atten = self.conv_atten(x)
        out = x * x_atten
        return out


class ContextPath(nn.Module):
    def __init__(self, backbone: dict, out_channels):
        super().__init__()
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.arm5 = AttentionRefinementModule(self.backbone.C5)
        self.arm4 = AttentionRefinementModule(self.backbone.C4)
        self.conv_head5 = ConvModule(self.backbone.C5, self.backbone.C4, 3, padding=1)
        self.conv_head4 = ConvModule(self.backbone.C4, out_channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        _, _, x3, x4, x5 = self.backbone(x)
        out = self.pool(x5)

        x5_atten = self.arm5(x5)
        out = x5_atten + out
        out = self.conv_head5(self.up(out))

        x4_atten = self.arm4(x4)
        out = x4_atten + out
        out = self.conv_head4(self.up(out))

        return out, x3, x4


class SpatialPath(nn.Module):
    def __init__(self, out_channels, channels=64):
        super().__init__()
        self.layer1 = ConvModule(3, channels, 7, stride=2, padding=3)
        self.layer2 = ConvModule(channels, channels, 3, stride=2, padding=1)
        self.layer3 = ConvModule(channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, 1)
        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvModule(out_channels, out_channels, 1, normalize=False),
            ConvModule(out_channels, out_channels, 1, normalize=False, activate=False),
            nn.Sigmoid()
        )

    def forward(self, x_cp, x_sp):
        x = torch.cat([x_cp, x_sp], dim=1)
        x = self.conv(x)
        x_atten = self.conv_atten(x)
        x_atten = x * x_atten
        out = x + x_atten
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, channels, n_classes):
        super().__init__()
        self.conv = ConvModule(in_channels, channels, 3, padding=1)
        self.seg_top = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        out = self.seg_top(x)
        return out


class BiSeNetV1(nn.Module):
    def __init__(self, backbone: dict, n_classes: int):
        super().__init__()
        self.context_path = ContextPath(backbone, 128)
        self.spatial_path = SpatialPath(128)
        self.ffm = FeatureFusionModule(256, 256)
        self.head = SegHead(256, 256, n_classes)
        self.aux_head = nn.ModuleList([
            SegHead(self.context_path.backbone.C3, 64, n_classes),
            SegHead(self.context_path.backbone.C4, 64, n_classes)
        ])

        self._init_weights()

        self.seg_loss = OHEMCELoss(ignore_index=255, reduction='mean')
        self.aux_loss = nn.ModuleList([
            OHEMCELoss(ignore_index=255, reduction='mean'),
            OHEMCELoss(ignore_index=255, reduction='mean')
        ])

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'context_path.backbone' in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    if 'seg_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x_cp, *auxs = self.context_path(x)
        x_sp = self.spatial_path(x)
        x = self.ffm(x_cp, x_sp)
        out = self.head(x)
        aux_outs = [self.aux_head[i](auxs[i]) for i in range(2)]
        return (out, *aux_outs)

    def parameters(self, cfg):
        base_lr = cfg.pop('lr')
        base_wd = cfg.pop('weight_decay', 0.0)
        param_groups = [
            {'params': [], 'lr': base_lr * 0.1, 'weight_decay': base_wd, **cfg},
            {'params': [], 'lr': base_lr * 0.1, 'weight_decay': 0.0, **cfg},
            {'params': [], 'lr': base_lr, 'weight_decay': base_wd, **cfg},
            {'params': [], 'lr': base_lr, 'weight_decay': 0.0, **cfg},
        ]
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'context_path.backbone' in name:
                    no = 0 if p.ndim != 1 else 1
                else:
                    no = 2 if p.ndim != 1 else 3
                param_groups[no]['params'].append(p)
        return param_groups

    def loss(self, outputs, target):
        out, *aux_outs = outputs
        out = self._resize(out, target.size()[1:])
        aux_outs = [self._resize(aux_out, target.size()[1:])
                    for aux_out in aux_outs]

        loss = self.seg_loss(out, target)
        for i in range(2):
            loss += self.aux_loss[i](aux_outs[i], target)

        return loss

    def predict(self, outputs, target):
        out, *_ = outputs
        out = self._resize(out, target.size()[1:])
        out = out.argmax(dim=1)
        return out

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
