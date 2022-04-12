import torch
import torch.nn as nn
from torch.optim import SGD

from simss.schedulers.polynomial_lr import PolynomialLR


def test_polynomial_lr():
    m = nn.Conv2d(3, 64, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = PolynomialLR(optimizer, max_iterations=100, power=0.9)

    for i in range(100):
        x = torch.rand(2, 3, 64, 64)
        t = torch.randint(0, 10, (2, 64, 64))
        y = m(x)
        loss = criterion(y, t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        for group in scheduler.optimizer.param_groups:
            assert group['lr'] == max(0.01 * (1 - (i+1)/100) ** 0.9, 0.0001)
