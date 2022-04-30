#!/bin/bash

gsutil cp gs://simss-data/cityscapes_ss.zip .
gsutil cp -r gs://simss-data/assets/* ./assets/
unzip cityscapes_ss.zip -d ./data
rm cityscapes_ss.zip

pip install torchmetrics==0.7.3
pip install tensorboard==2.8.0
pip install -e .
