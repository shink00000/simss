#!/bin/bash

gsutil cp gs://simss-data/cityscapes_ss.zip .
gsutil cp gs://simss-data/assets/mit_b2.pth ./assets
unzip -q cityscapes_ss.zip -d ./data
rm cityscapes_ss.zip

pip install torchmetrics==0.7.3
pip install ensorboard==2.8.0
pip install -e .
