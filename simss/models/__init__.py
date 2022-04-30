from .bisenetv1 import BiSeNetV1
from .segformer import SegFormer
from .unet import UNet

MODELS = {
    'BiSeNetV1': BiSeNetV1,
    'SegFormer': SegFormer,
    'UNet': UNet
}
