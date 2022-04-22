import torch
import torch.nn as nn


def nlc_to_nchw(x, h, w):
    n, _, c = x.shape
    return x.transpose(1, 2).view(n, c, h, w)


def nchw_to_nlc(x):
    return x.flatten(2).transpose(1, 2)


class OverlapPatchEmbeding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            patch_size,
            stride=stride,
            padding=patch_size//2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        h, w = x.size()[2:]
        x = nchw_to_nlc(x)
        out = self.norm(x)

        return out, h, w


class DropPath(nn.Module):
    def __init__(self, drop_path_rate):
        super().__init__()
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        if self.drop_path_rate == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)

        return x * random_tensor


class EfficientSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, reduce_ratio, drop_path_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        if reduce_ratio > 1:
            self.reduction = nn.Conv2d(embed_dim, embed_dim, reduce_ratio, stride=reduce_ratio)
            self.norm = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, x0, h, w):
        x_q = x

        if hasattr(self, 'reduction'):
            x_kv = nlc_to_nchw(x, h, w)
            x_kv = self.reduction(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        x, _ = self.attn(x_q, x_kv, x_kv, need_weights=False)
        out = x0 + self.drop_path(x)

        return out


class MixFFN(nn.Module):
    def __init__(self, embed_dim, drop_path_rate):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, x0, h, w):
        x = self.fc1(x)
        x = nlc_to_nchw(x, h, w)
        x = self.act(self.conv(x))
        x = nchw_to_nlc(x)
        x = self.fc2(x)
        out = x0 + self.drop_path(x)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, reduce_ratio, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention(embed_dim, n_heads, reduce_ratio, drop_path_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MixFFN(embed_dim, drop_path_rate)

    def forward(self, x, h, w):
        x = self.attn(self.norm1(x), x, h, w)
        out = self.ffn(self.norm2(x), x, h, w)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, in_channels, embed_dim, patch_size, stride, n_heads,
                 reduce_ratio, drop_path_rates):
        super().__init__()
        self.patch = OverlapPatchEmbeding(in_channels, embed_dim, patch_size, stride)
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim,
                n_heads,
                reduce_ratio,
                drop_path_rates[i]
            ) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, h, w = self.patch(x)
        for m in self.layers:
            x = m(x, h, w)
        x = self.norm(x)
        out = nlc_to_nchw(x, h, w)

        return out


class MiT(nn.Module):
    def __init__(self, scale: str = 'b3'):
        assert scale in ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')

        super().__init__()
        in_channels = 3
        n_layers = self._n_layers(scale)
        embed_dims = self._embed_dims(scale)
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        n_heads = [1, 2, 5, 8]
        reduce_ratios = [8, 4, 2, 1]
        dprs = torch.linspace(0, 0.1, sum(n_layers)).tolist()

        self.layers = nn.ModuleList([])
        for i in range(4):
            self.layers.append(
                TransformerBlock(
                    n_layers=n_layers[i],
                    in_channels=in_channels,
                    embed_dim=embed_dims[i],
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    n_heads=n_heads[i],
                    reduce_ratio=reduce_ratios[i],
                    drop_path_rates=dprs[sum(n_layers[:i]):sum(n_layers[:i+1])]
                )
            )
            in_channels = embed_dims[i]

        for i, embed_dim in enumerate(embed_dims, start=2):
            setattr(self, f'C{i}', embed_dim)

        # self._init_weights()
        self.load_state_dict(torch.load(f'./assets/mit_{scale}.pth'))

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=pow(1.0 / fan_out, 0.5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _n_layers(self, scale: str):
        return {
            'b0': [2, 2, 2, 2],
            'b1': [2, 2, 2, 2],
            'b2': [3, 4, 6, 3],
            'b3': [3, 4, 18, 3],
            'b4': [3, 8, 27, 3],
            'b5': [3, 6, 40, 3]
        }[scale]

    def _embed_dims(self, scale: str):
        return {
            'b0': [32, 64, 160, 256],
            'b1': [64, 128, 320, 512],
            'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512],
            'b4': [64, 128, 320, 512],
            'b5': [64, 128, 320, 512]
        }[scale]


if __name__ == '__main__':
    m = MiT('b3')
    x = torch.rand(2, 3, 128, 128)
    outs = m(x)
    for out in outs:
        print(out.shape)
    print(m.C2, m.C5)
