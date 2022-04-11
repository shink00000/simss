import torch
import torch.nn as nn
from random import uniform, random, choice
from torchvision import transforms as T
from torchvision.transforms import functional as F


class CopyPaste(nn.Module):
    def __init__(self, ignore_index=255, p=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.p = p

    def forward(self, data: tuple) -> tuple:
        image, label, image_copy, label_copy = data
        if random() < self.p:
            r = uniform(0.5, 1.0)
            h, w = image.size()[1:]
            size = rh, rw = int(r*h), int(r*w)
            ph, pw = int((h - rh)*random()), int((w - rw)*random())
            image_copy = F.resize(image_copy, size, interpolation=T.InterpolationMode.BILINEAR)
            label_copy = F.resize(label_copy, size, interpolation=T.InterpolationMode.NEAREST)
            if random() < 0.5:
                image_copy = image_copy.flip(2)
                label_copy = label_copy.flip(2)
            id = choice(label_copy[label_copy.ne(self.ignore_index)].unique())
            mask = label_copy == id
            image[:, ph:ph+rh, pw:pw+rw] = torch.where(mask, image_copy, image[:, ph:ph+rh, pw:pw+rw])
            label[:, ph:ph+rh, pw:pw+rw] = torch.where(mask, label_copy, label[:, ph:ph+rh, pw:pw+rw])
        return (image, label)


class RandomScaleCrop(nn.Module):
    def __init__(self, scales: list):
        super().__init__()
        self.scales = scales

    def forward(self, data: tuple) -> tuple:
        image, label = data
        scale = choice(self.scales)
        h, w = image.size()[1:]
        sh, sw = int(scale*h), int(scale*w)
        ch, cw = int((h - sh)*random()), int((w - sw)*random())
        image = image[:, ch:ch+sh, cw:cw+sw]
        label = label[:, ch:ch+sh, cw:cw+sw]
        return (image, label)


class Resize(nn.Module):
    def __init__(self, size: list):
        super().__init__()
        self.size = size

    def forward(self, data: tuple) -> tuple:
        image, label = data
        image = F.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        label = F.resize(label, self.size, interpolation=T.InterpolationMode.NEAREST)
        return (image, label)


class HorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, data: tuple) -> tuple:
        image, label = data
        if random() < self.p:
            image = image.flip(2)
            label = label.flip(2)
        return (image, label)


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self, data: tuple):
        image, label = data
        image = self.color_jitter(image)
        return (image, label)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.normalize = T.Normalize(mean, std)

    def forward(self, data: tuple):
        image, label = data
        image = self.normalize(image)
        return (image, label)


class Rotate(nn.Module):
    def __init__(self, degrees, fill=0, label_fill=255):
        super().__init__()
        self.degrees = degrees
        self.fill = fill
        self.label_fill = label_fill

    def forward(self, data: tuple):
        image, label = data
        angle = uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, fill=self.fill, interpolation=T.InterpolationMode.BILINEAR)
        label = F.rotate(label, angle, fill=self.label_fill)
        return (image, label)
