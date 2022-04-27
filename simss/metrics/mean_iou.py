import torch
from torchmetrics import Metric


class MeanIoU(Metric):
    """
    Inputs:
        y_true: (N, H, W)
        y_pred: (N, C, H, W) logits
    """

    def __init__(self, n_classes: int, class_names: list, ignore_index: int = 255):
        super().__init__()

        self.n_classes = n_classes
        self.class_names = class_names
        self.ignore_index = ignore_index

        self.add_state('confmat', default=torch.zeros(n_classes, n_classes), dist_reduce_fx='sum')

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_pred = y_pred.argmax(dim=1)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        valid = y_true != self.ignore_index
        y_pred, y_true = y_pred[valid], y_true[valid]

        count = (y_true * self.n_classes + y_pred).bincount(minlength=self.n_classes**2)
        confmat = count.reshape(self.n_classes, self.n_classes)
        self.confmat += confmat.cpu()

    def compute(self) -> dict:
        tp = self.confmat.diag()
        fp = self.confmat.sum(dim=1) - tp
        fn = self.confmat.sum(dim=0) - tp
        ious = torch.div(tp, tp + fp + fn).numpy()

        for i in range(self.n_classes):
            print(f'{self.class_names[i]:20}: {ious[i]:.04f}')
        print(f'{"mean":20}: {ious.mean():.04f}')

        return {'mIoU': ious.mean()}
