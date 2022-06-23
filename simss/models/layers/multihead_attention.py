import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dims = embed_dim
        self.n_heads = n_heads
        self.d = (embed_dim//n_heads) ** 0.5
        self.in_proj_q = nn.Linear(embed_dim, embed_dim)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): (N, L, C)
            k (torch.Tensor): (N, S, C)
            v (torch.Tensor): (N, S, C)
            attn_mask (torch.Tensor, optional): (N, H, L, S). Defaults to None.

        Returns:
            torch.Tensor: (N, L, C)
        """
        q = self.in_proj_q(q)
        k = self.in_proj_k(k)
        v = self.in_proj_v(v)

        q = self._split_head(q, self.n_heads)
        k = self._split_head(k, self.n_heads)
        v = self._split_head(v, self.n_heads)

        attn = torch.matmul(q / self.d, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self._combine_head(out)
        out = self.out_proj(out)

        return out

    @staticmethod
    def _split_head(x, h):
        n, l, c = x.shape
        return x.reshape(n, l, h, c//h).transpose(1, 2)

    @staticmethod
    def _combine_head(x):
        return x.transpose(1, 2).flatten(2)
