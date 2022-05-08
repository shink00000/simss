import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


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


class HRModule(nn.Module):
    def __init__(self, n_branches: int, block: nn.Module, n_channels: list):
        super().__init__()
        self.n_branches = n_branches
        self.branches = nn.ModuleList(
            [self._create_block(block, n_channels[i]) for i in range(n_branches)]
        )
        self.fuse_layers = nn.ModuleList([])
        for i in range(n_branches):
            layers = nn.ModuleList([])
            for j in range(n_branches):
                if i < j:
                    layer = nn.Sequential(
                        nn.Conv2d(n_channels[j], n_channels[i], 1, bias=False),
                        nn.BatchNorm2d(n_channels[i]),
                        nn.UpsamplingBilinear2d(scale_factor=2**(j-i))
                    )
                elif i == j:
                    layer = nn.Identity()
                elif i > j:
                    layers_ = []
                    for _ in range(i-j-1):
                        layers_.extend([
                            nn.Conv2d(n_channels[j], n_channels[j], 3, 2, 1, bias=False),
                            nn.BatchNorm2d(n_channels[j]),
                            nn.ReLU(inplace=True)
                        ])
                    layers_.extend([
                        nn.Conv2d(n_channels[j], n_channels[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(n_channels[i])
                    ])
                    layer = nn.Sequential(*layers_)
                layers.append(layer)
            self.fuse_layers.append(layers)

    def _create_block(self, block: nn.Module, channels: int) -> nn.Module:
        if block is Bottleneck:
            downsample = nn.Sequential(
                nn.Conv2d(channels, 4*channels, 1, bias=False),
                nn.BatchNorm2d(4*channels)
            )
            layers = [block(channels, channels, downsample=downsample)]
            for _ in range(3):
                layers.append(block(4*channels, channels))
        else:
            layers = [block(channels, channels) for _ in range(4)]
        return nn.Sequential(*layers)

    def forward(self, xs: list):
        for i in range(self.n_branches):
            xs[i] = self.branches[i](xs[i])
        outs = []
        for i in range(self.n_branches):
            x = 0
            for j in range(self.n_branches):
                x += self.fuse_layers[i][j](xs[j])
            outs.append(x)
        return outs


class HRNet(nn.Module):
    def __init__(self, width: int = 48, pretrain: str = None):
        assert width in (18, 32, 48)

        super().__init__()
        self.conv1 = ConvModule(3, 64, 3, 2)
        self.conv2 = ConvModule(64, 64, 3, 2)
        self.stage1 = nn.ModuleList([HRModule(1, Bottleneck, [64])])
        self.trans1 = nn.ModuleList([
            ConvModule(256, width, 3),
            ConvModule(256, width*2, 3, 2)
        ])
        self.stage2 = nn.ModuleList([HRModule(2, BasicBlock, [width, width*2])])
        self.trans2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            ConvModule(width*2, width*4, 3, 2)
        ])
        self.stage3 = nn.ModuleList([HRModule(3, BasicBlock, [width, width*2, width*4]) for _ in range(4)])
        self.trans3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            ConvModule(width*4, width*8, 3, 2)
        ])
        self.stage4 = nn.ModuleList([HRModule(4, BasicBlock, [width, width*2, width*4, width*8]) for _ in range(3)])

        for i in range(2, 6):
            setattr(self, f'C{i}', width * 2 ** (i-2))

        self._init_weights()
        if pretrain:
            self.load_state_dict(torch.load(pretrain))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)

        # stage1 ~ stage4
        xs = [x]
        for num in range(1, 4+1):
            ys = []
            for module in getattr(self, f'stage{num}'):
                xs = module(xs)
            if num < 4:
                for i in range(len(getattr(self, f'trans{num}'))):
                    y = getattr(self, f'trans{num}')[i](xs[min(i, len(xs)-1)])
                    ys.append(y)
                xs = ys

        return xs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=pow(1.0 / fan_out, 0.5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    m = HRNet(width=48)
    outs = m(x)
    for out in outs:
        print(out.shape)
