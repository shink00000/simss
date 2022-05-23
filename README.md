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

### SegFormer-SwinT [[arch](./docs/archs/segformer_swin-t.txt)]

- [config](./configs/segformer_swin-t_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: Swin-Tiny
- [tensorboard](https://tensorboard.dev/experiment/wTM78yraRx6jCrjJkDAqYg/)
- evaluation result
  ```
  road                : 0.9808
  sidewalk            : 0.8490
  building            : 0.9254
  wall                : 0.5843
  fence               : 0.6129
  pole                : 0.6300
  trafficlight        : 0.6986
  trafficsign         : 0.7973
  vegetation          : 0.9280
  terrain             : 0.6577
  sky                 : 0.9517
  person              : 0.8184
  rider               : 0.6001
  car                 : 0.9465
  truck               : 0.7001
  bus                 : 0.7930
  train               : 0.4739
  motorcycle          : 0.6612
  bicycle             : 0.7759
  *mean*              : 0.7571
  ```

## [Lawin](https://arxiv.org/abs/2201.01615)

### Lawin-MiTB2 [[arch](./docs/archs/lawin_mit-b2.txt)]

- notes
  - mIoU was about 2.5% lower than the evaluation results reported above (81.7%). I suspect the reason is that the input size was changed from (768, 768) -> (512, 1024) and that it was run on a single gpu training with batch_size = 4 (i.e. insufficient batch size).
  - The following points differ from the official implementation:
    - Conv2d channels in head (512 -> 128)
    - The use of multihead attention instead of nonlocal.
    - The official implementation divides the sum of maxpool and avgpool results by 2, but here only avgpool is used.
- [config](./configs/lawin_mit-bw_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: MiT-b2
- [tensorboard](https://tensorboard.dev/experiment/miPjAck7RK2WaEE5rFRIow/)
- evaluation result
  ```
  road                : 0.9813
  sidewalk            : 0.8483
  building            : 0.9267
  wall                : 0.5269
  fence               : 0.5976
  pole                : 0.6666
  trafficlight        : 0.7232
  trafficsign         : 0.8089
  vegetation          : 0.9292
  terrain             : 0.6349
  sky                 : 0.9525
  person              : 0.8321
  rider               : 0.6231
  car                 : 0.9545
  truck               : 0.8274
  bus                 : 0.8974
  train               : 0.8116
  motorcycle          : 0.7065
  bicycle             : 0.7908
  *mean*              : 0.7916
  ```

## [DeepLabV3+](https://arxiv.org/abs/1802.02611v3)

### DeepLabV3+-ResNet50 [[arch](./docs/archs/deeplabv3p_r50.txt)]

- [config](./configs/deeplabv3p_r50_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: ResNet50
- [tensorboard](https://tensorboard.dev/experiment/eGi7tk0wTrSpfEwTQiyGSw/)
- evaluation result
  ```
  road                : 0.9806
  sidewalk            : 0.8456
  building            : 0.9128
  wall                : 0.4353
  fence               : 0.5243
  pole                : 0.6211
  trafficlight        : 0.6872
  trafficsign         : 0.7712
  vegetation          : 0.9220
  terrain             : 0.6428
  sky                 : 0.9474
  person              : 0.7908
  rider               : 0.5030
  car                 : 0.9283
  truck               : 0.5157
  bus                 : 0.5181
  train               : 0.7140
  motorcycle          : 0.5558
  bicycle             : 0.7585
  *mean*              : 0.7144
  ```

## [OCRNet](https://arxiv.org/abs/1909.11065v6)

### OCRNet-HRNetV2W32 [[arch](./docs/archs/ocrnet_hr32.txt)]

- [config](./configs/ocrnet_hr32_cityscapes_h512_w1024.yaml)
  - data: CityScapes
  - input_size: (512, 1024)
  - backbone: HRNetV2W32
- [tensorboard](https://tensorboard.dev/experiment/VrJ4FgpZQiyETlTDe6PqgQ/)
- evaluation result
  ```
  road                : 0.9840
  sidewalk            : 0.8647
  building            : 0.9331
  wall                : 0.6203
  fence               : 0.6602
  pole                : 0.6838
  trafficlight        : 0.7209
  trafficsign         : 0.8046
  vegetation          : 0.9297
  terrain             : 0.6559
  sky                 : 0.9522
  person              : 0.8297
  rider               : 0.6178
  car                 : 0.9565
  truck               : 0.8622
  bus                 : 0.9129
  train               : 0.8045
  motorcycle          : 0.6980
  bicycle             : 0.7832
  *mean*              : 0.8039
  ```
