import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.utils import nchw_to_nlc, nlc_to_nchw
from .backbones import BACKBONES


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        out = self.act(x)

        return out


class ObjectRegionRepresentations(nn.Module):
    def forward(self, obj: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj (torch.Tensor): (N, n_classes, H, W)
            pix (torch.Tensor): (N, C, H, W)

        Returns:
            torch.Tensor: (N, C, n_classes, 1)
        """
        obj = nchw_to_nlc(obj)
        pix = nchw_to_nlc(pix)
        obj = F.softmax(obj, dim=1)
        out = torch.matmul(pix, obj.transpose(-2, -1)).unsqueeze(3)

        return out


class AugmentedRepresentations(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obj: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj (torch.Tensor): (N, C, n_classes, 1)
            pix (torch.Tensor): (N, C, H, W)

        Returns:
            torch.Tensor: (N, C, n_classes, 1)
        """
        pass


class OCRHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, drop_rate: float = 0.05):
        super().__init__()
        self.soft_object_regions = nn.Sequential(
            ConvModule(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, n_classes, 1)
        )
        self.pixel_reprs = ConvModule(in_channels, 512, 3)

        self.in_proj_q = nn.Sequential(
            ConvModule(512, 256, 1),
            ConvModule(256, 256, 1)
        )
        self.in_proj_k = nn.Sequential(
            ConvModule(512, 256, 1),
            ConvModule(256, 256, 1)
        )
        self.in_proj_v = ConvModule(512, 256, 1)
        self.out_proj = ConvModule(256, 512, 1)
        self.d = 256 ** 0.5

        self.aug_reprs = ConvModule(512*2, 512, 1)

        self.drop = nn.Dropout2d(drop_rate)
        self.seg_top = nn.Conv2d(512, n_classes, 1)

        self._init_weights()

    def forward(self, xs):
        for i in range(1, 4):
            xs[i] = self._resize(xs[i], size=xs[0].size()[2:])
        x = torch.cat(xs, dim=1)

        h, w = x.size()[2:]
        pix_rpr = self.pixel_reprs(x)  # (N, C, H, W)
        aux = self.soft_object_regions(x)  # (N, K, H, W)

        # Object Region Representations
        attn = F.softmax(nchw_to_nlc(aux), dim=1)  # (N, HxW, K)
        obj_rpr = torch.matmul(pix_rpr.flatten(2), attn).unsqueeze(3)  # (N, C, K, 1)

        # Object Contextual Representations
        q = nchw_to_nlc(self.in_proj_q(pix_rpr))  # (N, HxW, C)
        k = nchw_to_nlc(self.in_proj_k(obj_rpr))  # (N, K, C)
        v = nchw_to_nlc(self.in_proj_v(obj_rpr))  # (N, K, C)
        attn = torch.matmul(q / self.d, k.transpose(-2, -1))  # (N, HxW, K)
        attn = F.softmax(attn, dim=-1)
        v = torch.matmul(attn, v)  # (N, HxW, C)
        v = nlc_to_nchw(v, h, w)
        con_rpr = self.out_proj(v)

        # Augmented Representations
        aug_rpr = self.aug_reprs(torch.cat([con_rpr, pix_rpr], dim=1))

        x = self.drop(aug_rpr)
        out = self.seg_top(x)

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


class OCRNet(nn.Module):
    def __init__(self, backbone: dict, n_classes: int):
        super().__init__()
        self.encoder = BACKBONES[backbone.pop('type')](**backbone)
        self.decoder = OCRHead(
            in_channels=sum([self.encoder.C2, self.encoder.C3, self.encoder.C4, self.encoder.C5]),
            n_classes=n_classes
        )

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.aux_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, x):
        xs = self.encoder(x)
        out, aux = self.decoder(xs)

        return out, aux

    def get_param_groups(self, cfg):
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
