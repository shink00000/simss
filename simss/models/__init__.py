from .bisenetv1 import BiSeNetV1
from .segformer import SegFormer
from .lawin import Lawin
from .ocrnet import OCRNet
from .unet import UNet

MODELS = {
    'BiSeNetV1': BiSeNetV1,
    'SegFormer': SegFormer,
    'Lawin': Lawin,
    'OCRNet': OCRNet,
    'UNet': UNet
}
