import torch
import torch.nn as nn
from random import uniform, random, choice, randint
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


class RandomCrop(nn.Module):
    def __init__(self, crop_size: list, cat_max_ratio: float = 0.75, ignore_index: int = 255):
        super().__init__()
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def forward(self, data: tuple) -> tuple:
        image, label = data
        h, w = image.size()[1:]
        ch, cw = self.crop_size
        for _ in range(10):
            cy, cx = randint(0, h - ch), randint(0, w - cw)
            crop = label[:, cy:cy+ch, cx:cx+cw]
            counts = crop[crop != self.ignore_index].unique(return_counts=True)[1]
            if len(counts) > 1 and (counts.max()/counts.sum() < self.cat_max_ratio):
                break
        image = image[:, cy:cy+ch, cx:cx+cw]
        label = label[:, cy:cy+ch, cx:cx+cw]

        return (image, label)


class Resize(nn.Module):
    def __init__(self, size: list, ratio_range: list = [1.0, 1.0]):
        super().__init__()
        self.size = size
        self.ratio_range = ratio_range

    def forward(self, data: tuple) -> tuple:
        image, label = data
        ratio = uniform(*self.ratio_range)
        size = [int(v * ratio) for v in self.size]
        image = F.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
        label = F.resize(label, size, interpolation=T.InterpolationMode.NEAREST)
        return (image, label)


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, data: tuple) -> tuple:
        image, label = data
        if random() < self.p:
            image = image.flip(2)
            label = label.flip(2)
        return (image, label)


class PhotoMetricDistortion(nn.Module):
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, data: tuple):
        image, label = data
        mode = randint(0, 1)
        if mode == 0:
            image = self._adjust_contrast(image)
        image = self._adjust_brightness(image)
        image = self._rgb_to_hsv(image)
        image = self._adjust_saturation(image)
        image = self._adjust_hue(image)
        image = self._hsv_to_rgb(image)
        if mode == 1:
            image = self._adjust_contrast(image)
        return (image, label)

    def _adjust_brightness(self, image):
        if randint(0, 1):
            image = image + uniform(-self.brightness, self.brightness)
            image = image.clip(0, 1)
        return image

    def _adjust_contrast(self, image):
        if randint(0, 1):
            image = image * uniform(1-self.contrast, 1+self.contrast)
            image = image.clip(0, 1)
        return image

    def _adjust_saturation(self, image):
        if randint(0, 1):
            image[1] = image[1] * uniform(1-self.saturation, 1+self.saturation)
            image[1] = image[1].clip(0, 1)
        return image

    def _adjust_hue(self, image):
        if randint(0, 1):
            image[0] = image[0] + 180 * uniform(-self.hue, self.hue)
            image[0] = image[0] % 360
        return image

    def _rgb_to_hsv(self, image, eps=1e-8):
        # https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        r, g, b = image
        max_rgb, argmax_rgb = image.max(0)
        min_rgb, _ = image.min(0)

        v = max_rgb
        s = torch.where(v != 0, (v - min_rgb) / v, torch.zeros_like(v))
        h = torch.stack([
            60 * (g - b) / (v - min_rgb + eps),
            60 * (b - r) / (v - min_rgb + eps) + 120,
            60 * (r - g) / (v - min_rgb + eps) + 240
        ], dim=0).gather(dim=0, index=argmax_rgb[None]).squeeze(0) % 360

        return torch.stack([h, s, v], dim=0)

    def _hsv_to_rgb(self, image):
        # https://www.rapidtables.com/convert/color/hsv-to-rgb.html
        h, s, v = image
        c = v * s
        x = c * (1 - (h / 60 % 2 - 1).abs())
        m = v - c
        z = torch.zeros_like(c)
        h_id = (h / 60).long().clip(0, 5)
        r_ = torch.stack([c, x, z, z, x, c], dim=0).gather(dim=0, index=h_id[None])
        g_ = torch.stack([x, c, c, x, z, z], dim=0).gather(dim=0, index=h_id[None])
        b_ = torch.stack([z, z, x, c, c, x], dim=0).gather(dim=0, index=h_id[None])

        return torch.cat([r_ + m, g_ + m, b_ + m], dim=0)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.normalize = T.Normalize(mean, std)

    def forward(self, data: tuple):
        image, label = data
        image = self.normalize(image)
        return (image, label)


class RandomRotate(nn.Module):
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
