from .bisenetv1 import BiSeNetV1
from .segformer import SegFormer
from .lawin import Lawin
from .unet import UNet

MODELS = {
    'BiSeNetV1': BiSeNetV1,
    'SegFormer': SegFormer,
    'Lawin': Lawin,
    'UNet': UNet
}
