from .resnet import ResNet
from .mit import MixTransformer
from .swin import SwinTransformer


BACKBONES = {
    'ResNet': ResNet,
    'MixTransformer': MixTransformer,
    'SwinTransformer': SwinTransformer
}
