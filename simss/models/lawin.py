import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import nchw_to_nlc, nlc_to_nchw
from .segformer import ConvModule
from .backbones import BACKBONES


class PositionMixing(nn.Module):
    def __init__(self, patch_size: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.mixers = nn.ModuleList([nn.Linear(patch_size**2, patch_size**2) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, L, C)

        Returns:
            torch.Tensor: (N, L, C)
        """
        _, _, c = x.size()
        x = x.transpose(1, 2).contiguous()
        x_mlp = []
        for i in range(self.n_heads):
            x_cur = x[:, (c//self.n_heads)*i:(c//self.n_heads)*(i+1)]
            x_cur = self.mixers[i](x_cur)
            x_mlp.append(x_cur)
        x_mlp = torch.cat(x_mlp, dim=1)
        out = (x + x_mlp).transpose(1, 2).contiguous()

        return out


class LargeWindowAttention(nn.Module):
    def __init__(self, r: int, embed_dim: int, n_heads: int, patch_size: int = 8, window_stride: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.window_size = r * patch_size
        self.window_stride = window_stride

        self.pool = nn.AvgPool2d(r, r)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.position_mixing = PositionMixing(patch_size, n_heads)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, C, H, W)

        Returns:
            torch.Tensor: (N, C, H, W)
        """
        h, w = x.size()[2:]

        q = self.window_partition(x, self.patch_size, self.window_stride)
        q = nchw_to_nlc(q)

        c = self.window_partition(x, self.window_size, self.window_stride)
        c = self.pool(c)
        c = nchw_to_nlc(c)
        c = self.norm(c)
        c = self.position_mixing(c)

        x = self.attn(q, c, c, need_weights=False)[0]
        x = nlc_to_nchw(x, self.patch_size, self.patch_size)
        out = self.window_reverse(x, self.patch_size, h, w)

        return out

    @staticmethod
    def window_partition(x: torch.Tensor, window_size: int, window_stride: int = 8) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, C, H, W)
            window_size (int): window size
            window_stride (int, optional): window stride. Defaults to 8.

        Returns:
            torch.Tensor: (N*nP, C, wH, wW)
        """
        _, c, h, w = x.shape
        x = F.unfold(x, window_size, stride=window_stride, padding=(window_size-window_stride)//2)
        x = x.view(-1, c, window_size, window_size, h//window_stride, w//window_stride)
        out = x.permute(0, 4, 5, 1, 2, 3).contiguous().flatten(0, 2)  # (N*nP, C, wH, wW)

        return out

    @staticmethod
    def window_reverse(x: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N*nP, C, wH, wW)
            window_size (int): window size
            h (int): feature height
            w (int): feature width

        Returns:
            torch.Tensor: (N, C, H, W)
        """
        _, c, _, _ = x.shape
        x = x.view(-1, h//window_size, w//window_size, c, window_size, window_size)
        out = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, c, h, w)

        return out


class LawinHead(nn.Module):
    def __init__(self, in_channels: list, n_classes: int, drop_rate: float = 0.1):
        super().__init__()
        self.mlp_layers = nn.ModuleList([ConvModule(in_channels[i], 48 if i == 0 else 128, 1) for i in range(4)])
        self.mlp1 = ConvModule(3*128, 128, 1)
        self.lawin_attns = nn.ModuleList([LargeWindowAttention(r, 128, r**2) for r in [2, 4, 8]])
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(128, 128, 1)
        )
        self.mlp2 = ConvModule(5*128, 256, 1)
        self.mlp3 = ConvModule(256+48, 256, 1)
        self.drop = nn.Dropout2d(drop_rate)
        self.seg_top = nn.Conv2d(256, n_classes, 1)

        self._init_weights()

    def forward(self, xs):
        # upsample -> cat
        xs_ = []
        for i in range(1, 4):
            x_ = self.mlp_layers[i](xs[i])
            if i != 0:
                x_ = self._resize(x_, size=xs[1].size()[2:])
            xs_.append(x_)
        x2 = torch.cat(xs_, dim=1)
        x2 = self.mlp1(x2)

        # lawin ASPP
        xs_ = [x2]
        for i in range(3):
            x_ = self.lawin_attns[i](x2)
            xs_.append(x_)
        x_ = self._resize(self.image_pooling(x2), size=xs[1].size()[2:])
        xs_.append(x_)
        x2 = torch.cat(xs_, dim=1)
        x2 = self.mlp2(x2)

        # low level fuse
        x1 = self.mlp_layers[0](xs[0])
        x2 = self._resize(x2, size=x1.size()[2:])
        x = torch.cat([x1, x2], dim=1)
        x = self.mlp3(x)

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


class Lawin(nn.Module):
    def __init__(self, backbone: dict, n_classes: int):
        super().__init__()
        self.encoder = BACKBONES[backbone.pop('type')](**backbone)
        self.decoder = LawinHead(
            in_channels=[self.encoder.C2, self.encoder.C3, self.encoder.C4, self.encoder.C5],
            n_classes=n_classes
        )

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, x):
        xs = self.encoder(x)
        out = self.decoder(xs)

        return out

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
