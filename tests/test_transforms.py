from torchvision.io import read_image
import os
import os.path as osp
from PIL import Image

from simss.datasets import transforms as T


def load():
    dir_path = osp.join(osp.dirname(__file__), 'test_data')
    image = read_image(osp.join(dir_path, 'image1.png')) / 255
    label = read_image(osp.join(dir_path, 'label1.png')).long()
    data = (image, label)
    return data


def load_with_copy():
    dir_path = osp.join(osp.dirname(__file__), 'test_data')
    image = read_image(osp.join(dir_path, 'image1.png')) / 255
    label = read_image(osp.join(dir_path, 'label1.png')).long()
    image_copy = read_image(osp.join(dir_path, 'image2.png')) / 255
    label_copy = read_image(osp.join(dir_path, 'label2.png')).long()
    data = (image, label, image_copy, label_copy)
    return data


def save(image, label, file_name):
    dir_path = osp.join(osp.dirname(__file__), 'output')
    os.makedirs(dir_path, exist_ok=True)
    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    label = Image.fromarray(label.squeeze().numpy().astype('uint8'))
    merge = Image.new('RGB', size=(image.width, 2*image.height))
    merge.paste(image, (0, 0))
    merge.paste(label.convert('RGB'), (0, image.height))
    merge.save(osp.join(dir_path, f'{file_name}.png'))


def test_copy_paste():
    data = load_with_copy()
    t = T.CopyPaste(ignore_index=255, p=1.0)
    image, label = t(data)
    save(image, label, 'copy_paste')


def test_random_crop():
    data = load()
    for i in range(4):
        t = T.RandomCrop(crop_size=[512, 1024])
        image, label = t(data)
        save(image, label, f'random_crop_{i}')


def test_resize():
    data = load()
    t = T.Resize([448, 224])
    image, label = t(data)
    save(image, label, 'resize')


def test_horizontal_flip():
    data = load()
    t = T.RandomHorizontalFlip()
    image, label = t(data)
    save(image, label, 'horizontal_flip')


def test_photo_metric_distortion():
    data = load()
    for i in range(4):
        t = T.PhotoMetricDistortion()
        image, label = t(data)
        save(image, label, f'photo_{i}')


def test_normalize():
    data = load()
    t = T.Normalize()
    image, label = t(data)
    save(image, label, 'normalize')


def test_rotate():
    data = load()
    t = T.RandomRotate(degrees=20)
    image, label = t(data)
    save(image, label, 'rotate')
