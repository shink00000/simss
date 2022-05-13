import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
import os.path as osp
# from random import randint

from . import transforms as T


class CityScapesDataset(Dataset):
    CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight', 'trafficsign',
                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                   'train', 'motorcycle', 'bicycle']
    N_CLASSES = 19

    def __init__(self, data_dir: str, phase: str, size: list):
        super().__init__()
        # self.phase = phase
        self.data_list = self._get_data_list(data_dir, phase)
        self.transforms = self._get_transforms(phase, size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, label = self._load(*self.data_list[idx])
        data = (image, label)
        # if self.phase == 'train':
        #     idx = randint(0, len(self)-1)
        #     image_copy, label_copy = self._load(*self.data_list[idx])
        #     data = (*data, image_copy, label_copy)
        image, label = self.transforms(data)
        label = label.squeeze()
        return image, label

    def _load(self, image_path: str, label_path: str) -> tuple:
        image = read_image(image_path) / 255
        label = read_image(label_path).long()
        return image, label

    def _get_data_list(self, data_dir, phase):
        data_list = []
        if phase == 'test':
            phase = 'val'
        with open(osp.join(data_dir, f'{phase}.txt'), 'r') as f:
            pathlist = f.readlines()
        for line in pathlist:
            image_path, label_path = line.strip().split(' ')
            image_path = osp.join(data_dir, image_path)
            label_path = osp.join(data_dir, label_path)
            if osp.exists(image_path) and osp.exists(label_path):
                data_list.append((image_path, label_path))
        return data_list

    def _get_transforms(self, phase, size) -> nn.Sequential:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if phase == 'train':
            transforms = nn.Sequential(
                # T.CopyPaste(ignore_index=255, p=0.5),
                T.Resize(size=[1024, 2048], ratio_range=[0.5, 2.0]),
                T.RandomCrop(crop_size=size, ignore_index=255),
                T.HorizontalFlip(p=0.5),
                T.PhotoMetricDistortion(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.1),
                T.Normalize(mean=mean, std=std),
                # T.Rotate(degrees=10)
            )
        elif phase == 'val':
            transforms = nn.Sequential(
                T.Resize(size=size),
                T.Normalize(mean=mean, std=std)
            )
        else:
            transforms = nn.Sequential(
                T.Normalize(mean=mean, std=std)
            )
        return transforms
