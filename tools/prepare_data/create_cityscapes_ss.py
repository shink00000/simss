"""
before doing this, need to do the following:
1. clone https://github.com/mcordts/cityscapesScripts.git
2. run cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
CITYSCAPES_DATASET=xxxxxx python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
"""

from pathlib import Path
from shutil import copy
from tqdm import tqdm
import json

src_dir = Path('./data/cityscapes')
dst_dir = Path('./data/cityscapes_ss')

for phase in ['train', 'val']:
    images = {}
    labels = {}
    for src_p in tqdm((src_dir/f'leftImg8bit/{phase}').glob('**/*.png')):
        images[src_p.name[:-16]] = src_p.relative_to(src_dir).as_posix()
        dst_p = dst_dir/src_p.relative_to(src_dir)
        dst_p.parent.mkdir(exist_ok=True, parents=True)
        copy(src_p, dst_p)
    for src_p in tqdm((src_dir/f'gtFine/{phase}').glob('**/*_labelTrainIds.png')):
        labels[src_p.name[:-25]] = src_p.relative_to(src_dir).as_posix()
        dst_p = dst_dir/src_p.relative_to(src_dir)
        dst_p.parent.mkdir(exist_ok=True, parents=True)
        copy(src_p, dst_p)
    assert images.keys() == labels.keys()
    path_list = [f'{images[key]} {labels[key]}\n' for key in sorted(images.keys())]
    with open(dst_dir/f'{phase}.txt', 'w') as f:
        f.writelines(path_list)

with open(dst_dir/'labelmap.json', 'w') as f:
    json.dump({
        '0': 'road',
        '1': 'sidewalk',
        '2': 'building',
        '3': 'wall',
        '4': 'fence',
        '5': 'pole',
        '6': 'trafficlight',
        '7': 'trafficsign',
        '8': 'vegetation',
        '9': 'terrain',
        '10': 'sky',
        '11': 'person',
        '12': 'rider',
        '13': 'car',
        '14': 'truck',
        '15': 'bus',
        '16': 'train',
        '17': 'motorcycle',
        '18': 'bicycle'
    }, f, indent=4)
