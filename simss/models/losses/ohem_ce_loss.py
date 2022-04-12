import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMCELoss(nn.CrossEntropyLoss):
    """ Cross Entropy Loss with Online Hard Example Mining

    Args:
        input (torch.Tensor): [N, C, H, W]
        target (torch.Tensor): [N, H, W]

    Examples:
        >>> criterion = OHEMCELoss(reduction='mean', ohem_thresh=0.7)
        >>> input = torch.rand(N, C, H, W)
        >>> target = torch.randint(0, C, (N, H, W))
        >>> loss = criterion(input, target)
    """

    def __init__(self, *args, ohem_thresh: float = 0.7, **kwargs):
        self.ohem_thresh = ohem_thresh
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().clone()
        valid = target != self.ignore_index
        target[~valid] = 0
        score = F.softmax(input, dim=1).gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        score = score.masked_fill(~valid, 1.0)
        min_kept = score.numel()//16
        hard_example_thresh = score.reshape(-1).topk(min_kept, largest=False)[0][-1].clip(min=self.ohem_thresh)

        ignore_mask = score > hard_example_thresh
        target[ignore_mask] = self.ignore_index

        return super().forward(input, target)
