import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWS(nn.Conv2d):
    def forward(self, x):
        weight_mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_std = self.weight.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
        weight = torch.div(self.weight - weight_mean, weight_std + 1e-5)
        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
