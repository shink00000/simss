import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones import MiT
from .losses import OHEMCELoss


class SegFormerHead(nn.Module):
    def __init__(self, in_channels: list, channels: int, n_classes: int, drop_rate: float = 0.1):
        super().__init__()
        self.mlp_layers = nn.ModuleList([nn.Conv2d(in_channels[i], channels, 1) for i in range(4)])
        self.mlp = nn.Sequential(
            nn.Conv2d(4*channels, channels, 1),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout2d(drop_rate)
        self.seg_top = nn.Conv2d(channels, n_classes, 1)

        self._init_weights()

    def forward(self, xs):
        xs_ = []
        for i in range(4):
            x = self.mlp_layers[i](xs[i])
            if i != 0:
                x = self._resize(x, size=xs[0].size()[2:])
            xs_.append(x)
        x = torch.cat(xs_, dim=1)
        x = self.mlp(x)
        x = self.drop(x)
        out = self.seg_top(x)

        return out

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    if 'seg_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)


class SegFormer(nn.Module):
    def __init__(self, scale: str, n_classes: int):
        super().__init__()
        self.encoder = MiT(scale)
        self.decoder = SegFormerHead(
            in_channels=[self.encoder.C2, self.encoder.C3, self.encoder.C4, self.encoder.C5],
            channels=self._channels(scale),
            n_classes=n_classes
        )

        self.seg_loss = OHEMCELoss(ignore_index=255, reduction='mean')
        self.aux_loss = nn.ModuleList([
            OHEMCELoss(ignore_index=255, reduction='mean'),
            OHEMCELoss(ignore_index=255, reduction='mean')
        ])

    def forward(self, x):
        xs = self.encoder(x)
        out = self.decoder(xs)

        return out

    def parameters(self, cfg):
        param_groups = [
            {'params': [], 'weight_decay': 0.0},
            {'params': []}
        ]
        for name, p in self.named_parameters():
            if 'norm' in name:
                param_groups[0]['params'].append(p)
            else:
                param_groups[1]['params'].append(p)

        return param_groups

    def loss(self, output, target):
        output = self._resize(output, target.size()[1:])
        loss = self.seg_loss(output, target)

        return loss

    def predict(self, output, target):
        output = self._resize(output, target.size()[1:])
        output = output.argmax(dim=1)

        return output

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def _channels(self, scale):
        return {
            'b0': 256,
            'b1': 256,
            'b2': 768,
            'b3': 768,
            'b4': 768,
            'b5': 768
        }[scale]
