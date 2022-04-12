import torch
from torchmetrics import Metric


class MeanIoU(Metric):
    """
    Inputs:
        y_true: (N, H, W)
        y_pred: (N, H, W)
    """

    def __init__(self, n_classes: int, ignore_index: int = -1):
        super().__init__()

        self.n_classes = n_classes
        self.ignore_index = ignore_index

        self.add_state('cm', default=torch.zeros(n_classes, n_classes), dist_reduce_fx='sum')

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        valid = y_true != self.ignore_index
        y_pred, y_true = y_pred[valid], y_true[valid]

        count = (y_true * self.n_classes + y_pred).bincount(minlength=self.n_classes**2)
        cm = count.reshape(self.n_classes, self.n_classes)
        self.cm += cm.cpu()

    def compute(self) -> dict:
        tp = self.cm.diag()
        fp = self.cm.sum(dim=1) - tp
        fn = self.cm.sum(dim=0) - tp
        ious = torch.div(tp, tp + fp + fn).numpy()

        return ious.mean()
