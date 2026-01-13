# HaMeR: Hand Mesh Recovery

Forked from the official HaMeR repository: https://github.com/geopavlakos/hamer

Also incorporates changes from Dyn-HaMR’s fork: https://github.com/ZhengdiYu/hamer



## Installation


```bash
conda create -n dynhamr python=3.12 -y
conda activate dynhamr

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install git+https://github.com/facebookresearch/detectron2 --no-build-isolation

pip install ultralytics==8.1.34

pip install git+https://github.com/mattloper/chumpy --no-build-isolation

pip install -e .
```

~~Install ViTPose~~

We no longer need ViTPose.

```bash
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose

# See https://github.com/cocodataset/cocoapi/issues/141#issuecomment-386606299
pip install cython
pip install -e . --no-build-isolation
```