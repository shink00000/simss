from .resnet import ResNet
from .mit import MixTransformer
from .swin import SwinTransformer
from .hrnet import HRNet


BACKBONES = {
    'ResNet': ResNet,
    'MixTransformer': MixTransformer,
    'SwinTransformer': SwinTransformer,
    'HRNet': HRNet
}
