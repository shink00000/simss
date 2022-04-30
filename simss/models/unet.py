import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        out = self.act(x)

        return out


class ContractingStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvModule(in_channels, out_channels, 3),
            ConvModule(out_channels, out_channels, 3)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = x_cat = self.layers(x)
        out = self.pool(x)

        return out, x_cat


class ExpandingStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvModule(in_channels, out_channels, 3),
            ConvModule(out_channels, out_channels, 3)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 2),
            nn.ConstantPad2d([0, 1, 0, 1], 0)
        )

    def forward(self, x, x_cat):
        x = self.up(x)
        x = torch.cat([x_cat, x], dim=1)
        out = self.layers(x)

        return out


class UNet(nn.Module):
    def __init__(self, n_classes: int, channels: int = 64):
        super().__init__()
        self.contracting_path = nn.ModuleList([
            ContractingStage(3, channels*2**0),
            ContractingStage(channels*2**0, channels*2**1),
            ContractingStage(channels*2**1, channels*2**2),
            ContractingStage(channels*2**2, channels*2**3),
            ContractingStage(channels*2**3, channels*2**4)
        ])
        self.expanding_path = nn.ModuleList([
            ExpandingStage(channels*2**4, channels*2**3),
            ExpandingStage(channels*2**3, channels*2**2),
            ExpandingStage(channels*2**2, channels*2**1),
            ExpandingStage(channels*2**1, channels*2**0)
        ])
        self.seg_top = nn.Conv2d(channels, n_classes, 1)

        self._init_weights()

        self.seg_loss = nn.CrossEntropyLoss(
            weight=1/torch.tensor([
                0.36880975774750857,
                0.06086874054102551,
                0.22819570283385351,
                0.006559296959191458,
                0.008782959202080073,
                0.012275657265534682,
                0.0020849098115345898,
                0.0055286245692937885,
                0.15917384442191498,
                0.011586649487463331,
                0.04011514879508137,
                0.0121730286567891,
                0.0013485701801470244,
                0.07001107820593705,
                0.002676343508357241,
                0.002354059053386268,
                0.0023301726794817553,
                0.0009864704527446806,
                0.004138985628675025
            ]),
            ignore_index=255, reduction='mean')

    def forward(self, x):
        x_cats = []

        for layer in self.contracting_path:
            x, x_cat = layer(x)
            x_cats.append(x_cat)

        x = x_cats.pop()
        for layer in self.expanding_path:
            x_cat = x_cats.pop()
            x = layer(x, x_cat)
        out = self.seg_top(x)

        return out

    def parameters(self, cfg):
        param_groups = [
            {'params': [], 'weight_decay': 0.0},
            {'params': []}
        ]
        for name, p in self.named_parameters():
            if p.requires_grad:
                no = 0 if p.ndim == 1 else 1
                param_groups[no]['params'].append(p)

        return param_groups

    def loss(self, output, target):
        loss = self.seg_loss(output, target)

        return loss

    def predict(self, output, target):
        output = F.softmax(output, dim=1)

        return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    if 'seg_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)
