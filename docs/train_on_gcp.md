# train on gcp

## run VM instance
1. Menu -> Compute Engine -> VM instance
1. create instance -> Marketplace
1. search 'pytorch' and select 'Deep Learning VM'
1. start
1. Machine Family -> GPU
1. Enter other items

## prepare data and set environment
```bash
git clone https://github.com/shink00000/simss.git
cd simss

gsutil cp gs://simss-data/cityscapes_ss.zip .
gsutil cp gs://simss-data/assets/mit_b2.pth ./assets
unzip cityscapes_ss.zip -d ./data
rm cityscapes_ss.zip

pip install torchmetrics==0.7.3
pip install tensorboard==2.8.0
pip install -e .
```