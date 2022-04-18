import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import os.path as osp

from simss.schedulers.polynomial_lr import PolynomialLR


def test_polynomial_lr():
    m = nn.Conv2d(3, 64, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = PolynomialLR(optimizer, max_iterations=1000, power=0.9, warmup_iterations=100)

    lrs = []
    for i in range(1000):
        x = torch.rand(2, 3, 64, 64)
        t = torch.randint(0, 10, (2, 64, 64))
        y = m(x)
        loss = criterion(y, t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    plt.figure()
    plt.plot(lrs)
    plt.savefig(osp.join(osp.dirname(__file__), 'output/polynomial_lr.png'))
