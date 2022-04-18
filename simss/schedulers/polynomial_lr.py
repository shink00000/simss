from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iterations, power, min_lr=0.0001, last_epoch=-1, verbose=False):
        self.max_iterations = max_iterations
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        factor = max(1 - self.last_epoch / self.max_iterations, 0) ** self.power
        return [max(group['initial_lr'] * factor, self.min_lr) for group in self.optimizer.param_groups]
