from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iterations, power, min_lr=1e-4, warmup_iterations=1000,
                 last_epoch=-1, verbose=False):
        self.max_iterations = max_iterations
        self.power = power
        self.min_lr = min_lr
        self.warmup_iterations = warmup_iterations
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_iterations:
            factor = 0.9 * self.last_epoch / self.warmup_iterations + 0.1
            return [group['initial_lr'] * factor for group in self.optimizer.param_groups]
        else:
            last_epoch = self.last_epoch - self.warmup_iterations
            max_iterations = self.max_iterations - self.warmup_iterations
            factor = max(1 - last_epoch / max_iterations, 0) ** self.power
            return [max(group['initial_lr'] * factor, self.min_lr) for group in self.optimizer.param_groups]
