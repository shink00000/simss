import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import MiT


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        out = self.act(x)

        return out


class SegFormerHead(nn.Module):
    def __init__(self, in_channels: list, n_classes: int, drop_rate: float = 0.1):
        super().__init__()
        self.mlp_layers = nn.ModuleList([ConvModule(in_channels[i], 256, 1) for i in range(4)])
        self.mlp = ConvModule(4*256, 256, 1)
        self.drop = nn.Dropout2d(drop_rate)
        self.seg_top = nn.Conv2d(256, n_classes, 1)

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
                if 'seg_top' in name:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)


class SegFormer(nn.Module):
    def __init__(self, scale: str, n_classes: int, pretrain: str):
        super().__init__()
        self.encoder = MiT(scale, pretrain)
        self.decoder = SegFormerHead(
            in_channels=[self.encoder.C2, self.encoder.C3, self.encoder.C4, self.encoder.C5],
            n_classes=n_classes
        )

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, x):
        xs = self.encoder(x)
        out = self.decoder(xs)

        return out

    def parameters(self, cfg):
        base_lr = cfg['lr']
        param_groups = [
            {'params': [], 'lr': base_lr * 0.1, 'weight_decay': 0.0},
            {'params': [], 'lr': base_lr * 0.1},
            {'params': []}
        ]
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'encoder' in name:
                    no = 0 if 'norm' in name else 1
                else:
                    no = 2
                param_groups[no]['params'].append(p)

        return param_groups

    def loss(self, output, target):
        output = self._resize(output, target.size()[1:])
        loss = self.seg_loss(output, target)

        return loss

    def predict(self, output, target):
        output = self._resize(output, target.size()[1:])
        output = F.softmax(output, dim=1)

        return output

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)
