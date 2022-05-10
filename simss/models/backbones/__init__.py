from .resnet import ResNet
from .mit import MixTransformer
from .swin import SwinTransformer
from .hrnet import HRNetV2


BACKBONES = {
    'ResNet': ResNet,
    'MixTransformer': MixTransformer,
    'SwinTransformer': SwinTransformer,
    'HRNetV2': HRNetV2
}
