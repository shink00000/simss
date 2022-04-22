from .resnet import ResNet
from .mit import MiT


BACKBONES = {
    'ResNet': ResNet,
    'MiT': MiT
}
