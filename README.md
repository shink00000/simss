# SimSS
This repository will reproduce and implement well-known SS models.

# Policy
1. Simple implementation with less code and fewer files
1. Emphasis on processing efficiency
1. Be aware of ease of understanding

# Directory Layout
```
configs
  bisenetv1_r18_cityscapes_h512_w1024.yaml
simss
  datasets
    transforms.py
    cityscapes.py
    ...
  models
    backbones
      resnet.py
    losses
      ohem_ce_loss.py
      ...
    bisenetv1.py
  metrics
    mean_iou.py
    pixel_acc.py
  schedulers
    polynomial_lr.py
tools
  train.py
  test.py
```
