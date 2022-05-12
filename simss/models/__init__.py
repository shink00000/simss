from .bisenetv1 import BiSeNetV1
from .deeplabv3p import DeepLabV3P
from .segformer import SegFormer
from .lawin import Lawin
from .ocrnet import OCRNet
from .unet import UNet

MODELS = {
    'BiSeNetV1': BiSeNetV1,
    'DeepLabV3P': DeepLabV3P,
    'SegFormer': SegFormer,
    'Lawin': Lawin,
    'OCRNet': OCRNet,
    'UNet': UNet
}
