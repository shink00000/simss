import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BACKBONES
from .losses import OHEMCELoss


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activate: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if activate:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'act'):
            out = self.act(x)
        else:
            out = x
        return out


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, 3, padding=1)
        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvModule(out_channels, out_channels, kernel_size=1, activate=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x_atten = self.conv_atten(x)
        out = x * x_atten
        return out


class ContextPath(nn.Module):
    def __init__(self, backbone: dict, out_channels):
        super().__init__()
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.arm5 = AttentionRefinementModule(self.backbone.C5, out_channels)
        self.arm4 = AttentionRefinementModule(self.backbone.C4, out_channels)
        self.conv_head5 = ConvModule(out_channels, out_channels, 3, padding=1)
        self.conv_head4 = ConvModule(out_channels, out_channels, 3, padding=1)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvModule(self.backbone.C5, out_channels, 1)
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        _, _, _, x4, x5 = self.backbone(x)
        x5_gap = self.gap(x5)

        x5_atten = self.arm5(x5)
        x5 = x5_atten + x5_gap
        x5 = self.conv_head5(self.up(x5))

        x4_atten = self.arm4(x4)
        x4 = x4_atten + x5
        x4 = self.conv_head4(self.up(x4))

        return x4, x5


class SpatialPath(nn.Module):
    def __init__(self, out_channels, channels=64):
        super().__init__()
        self.layer1 = ConvModule(3, channels, 7, stride=2, padding=3)
        self.layer2 = ConvModule(channels, channels, 3, stride=2, padding=1)
        self.layer3 = ConvModule(channels, channels, 3, stride=2, padding=1)
        self.layer4 = ConvModule(channels, out_channels, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, 1)
        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvModule(out_channels, out_channels, 1),
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
    def __init__(self, in_channels, channels, n_classes, drop=0.0):
        super().__init__()
        self.conv = ConvModule(in_channels, channels, 3, padding=1)
        if drop > 0:
            self.drop = nn.Dropout2d(drop)
        self.seg_top = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'drop'):
            x = self.drop(x)
        out = self.seg_top(x)
        return out


class BiSeNetV1(nn.Module):
    def __init__(self, backbone: dict, n_classes: int):
        super().__init__()
        self.context_path = ContextPath(backbone, 128)
        self.spatial_path = SpatialPath(128)
        self.ffm = FeatureFusionModule(256, 256)
        self.head = SegHead(256, 256, n_classes, drop=0.1)
        self.aux_head = nn.ModuleList([
            SegHead(128, 64, n_classes),
            SegHead(128, 64, n_classes)
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
                if 'seg_top' in name:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x4, x5 = self.context_path(x)
        x_cp, *auxs = x4, x4, x5
        x_sp = self.spatial_path(x)
        x = self.ffm(x_cp, x_sp)
        out = self.head(x)
        auxs = [self.aux_head[i](auxs[i]) for i in range(2)]
        return (out, *auxs)

    def get_param_groups(self, cfg):
        return self.parameters()

    def loss(self, output, target):
        out, *auxs = output
        out = self._resize(out, target.size()[1:])
        auxs = [self._resize(aux, target.size()[1:])
                for aux in auxs]

        loss = self.seg_loss(out, target)
        for i in range(2):
            loss += self.aux_loss[i](auxs[i], target)

        return loss

    def predict(self, output, target):
        out, *_ = output
        out = self._resize(out, target.size()[1:])
        out = F.softmax(out, dim=1)
        return out

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)
