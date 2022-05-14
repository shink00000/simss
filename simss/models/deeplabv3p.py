import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BACKBONES


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        out = self.act(x)

        return out


class DepthwiseSeparatableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)

        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, out_channels, 1)
        )
        self.atrous_convs = nn.ModuleList([
            ConvModule(in_channels, out_channels, 1),
            DepthwiseSeparatableConvModule(in_channels, out_channels, 3, padding=12, dilation=12),
            DepthwiseSeparatableConvModule(in_channels, out_channels, 3, padding=24, dilation=24),
            DepthwiseSeparatableConvModule(in_channels, out_channels, 3, padding=36, dilation=36)
        ])
        self.fuse_conv = ConvModule(5*out_channels, out_channels, 1)

    def forward(self, x):
        xs = [self._resize(self.pool(x), x.size()[2:])]
        for i in range(4):
            xs.append(self.atrous_convs[i](x))
        x = torch.cat(xs, dim=1)
        out = self.fuse_conv(x)

        return out

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)


class DeepLabV3PHead(nn.Module):
    def __init__(self, in_channels: list, n_classes: int, drop_rate: float = 0.1):
        super().__init__()
        self.low_level = ConvModule(in_channels[0], 48, 1)
        self.aspp = ASPP(in_channels[2], 512)
        self.fuse_conv = nn.Sequential(
            DepthwiseSeparatableConvModule(48 + 512, 512, 3, padding=1),
            DepthwiseSeparatableConvModule(512, 512, 3, padding=1)
        )
        self.drop = nn.Dropout2d(drop_rate)
        self.seg_top = nn.Conv2d(512, n_classes, 1)
        self.aux_top = nn.Sequential(
            ConvModule(in_channels[1], 256, 3, padding=1),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, xs: list):
        _, x2, _, x4, x5 = xs
        x2 = self.low_level(x2)
        x5 = self._resize(self.aspp(x5), x2.size()[2:])
        x = torch.cat([x2, x5], dim=1)
        x = self.fuse_conv(x)
        x = self.drop(x)
        out = self.seg_top(x)
        aux = self.aux_top(x4)

        return out, aux

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'seg_top' in name:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)


class DeepLabV3P(nn.Module):
    def __init__(self, backbone: dict, n_classes: int):
        super().__init__()
        self.encoder = BACKBONES[backbone.pop('type')](**backbone)
        self.decoder = DeepLabV3PHead(
            in_channels=[self.encoder.C2, self.encoder.C4, self.encoder.C5],
            n_classes=n_classes
        )

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.aux_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, x):
        xs = self.encoder(x)
        out, aux = self.decoder(xs)

        return out, aux

    def parameters(self, cfg):
        base_lr = cfg['lr']
        param_groups = [
            {'params': [], 'lr': base_lr * 0.1, 'weight_decay': 0.0},
            {'params': [], 'lr': base_lr * 0.1},
            {'params': [], 'weight_decay': 0.0},
            {'params': []},
        ]
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'encoder' in name:
                    no = 0 if p.ndim == 1 else 1
                else:
                    no = 2 if p.ndim == 1 else 3
                param_groups[no]['params'].append(p)

        return param_groups

    def loss(self, output, target):
        out, aux = output
        out = self._resize(out, target.size()[1:])
        aux = self._resize(aux, target.size()[1:])
        loss = self.seg_loss(out, target) + 0.4 * self.aux_loss(aux, target)

        return loss

    def predict(self, output, target):
        out, _ = output
        out = self._resize(out, target.size()[1:])
        out = F.softmax(out, dim=1)

        return out

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)
