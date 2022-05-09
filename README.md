# SimSS

This repository will reproduce and implement well-known SS models.

# Policy

1. Simple implementation with less code and fewer files
1. Emphasis on processing efficiency
1. Be aware of ease of understanding

# Library Features

- [train](./tools/train.py)
- [test (evaluate)](./tools/test.py)

# Results

## [SegFormer](https://arxiv.org/abs/2105.15203)

### SegFormer-B0 [[arch](./docs/archs/segformer_mit-b0.txt)]

- notes
  - I referred to many of them: [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/mit.py)
  - mIoU was about 3% lower than the evaluation results reported above. I suspect the reason is that the input size was changed from (1024, 1024) -> (512, 1024) and that it was run on a single gpu training with batch_size = 4 (i.e. insufficient batch size).
- [config](./configs/segformer_mit-b0_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: MiT-b0
- [tensorboard](https://tensorboard.dev/experiment/jZM1DMx3RaKGspfAeq7psA/)
- evaluation result
  ```
  road                : 0.9761
  sidewalk            : 0.8139
  building            : 0.9102
  wall                : 0.4931
  fence               : 0.5209
  pole                : 0.5976
  trafficlight        : 0.6537
  trafficsign         : 0.7536
  vegetation          : 0.9215
  terrain             : 0.6323
  sky                 : 0.9410
  person              : 0.7917
  rider               : 0.5468
  car                 : 0.9404
  truck               : 0.7307
  bus                 : 0.7677
  train               : 0.6185
  motorcycle          : 0.6099
  bicycle             : 0.7567
  *mean*              : 0.7356
  ```

### SegFormer-B2 [[arch](./docs/archs/segformer_mit-b2.txt)]

- [config](./configs/segformer_mit-b2_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: MiT-b2
- [tensorboard](https://tensorboard.dev/experiment/GYcrBXvOT16GU1Pv10CbQQ/)
- evaluation result
  ```
  road                : 0.9832
  sidewalk            : 0.8585
  building            : 0.9270
  wall                : 0.5622
  fence               : 0.5937
  pole                : 0.6665
  trafficlight        : 0.7280
  trafficsign         : 0.8054
  vegetation          : 0.9290
  terrain             : 0.6367
  sky                 : 0.9515
  person              : 0.8305
  rider               : 0.6167
  car                 : 0.9546
  truck               : 0.8337
  bus                 : 0.8886
  train               : 0.7669
  motorcycle          : 0.7199
  bicycle             : 0.7891
  *mean*              : 0.7917
  ```
