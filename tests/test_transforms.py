from torchvision.io import read_image
import os.path as osp
from PIL import Image

from simss.datasets import transforms as T


def test_copy_paste():
    dir_path = osp.join(osp.dirname(__file__), 'test_data')
    image = read_image(osp.join(dir_path, 'image1.png')) / 255
    label = read_image(osp.join(dir_path, 'label1.png')).long()
    image_copy = read_image(osp.join(dir_path, 'image2.png')) / 255
    label_copy = read_image(osp.join(dir_path, 'label2.png')).long()
    data = (image, label, image_copy, label_copy)

    t = T.CopyPaste(ignore_index=255, p=0.5)
    image, label = t(data)

    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    label = Image.fromarray(label.squeeze().numpy().astype('uint8'))
    image.save(osp.join(dir_path, 'copy_paste_image.png'))
    label.save(osp.join(dir_path, 'copy_paste_label.png'))
