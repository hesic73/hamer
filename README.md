# HaMeR: Hand Mesh Recovery

Forked from the official HaMeR repository: https://github.com/geopavlakos/hamer

Also incorporates changes from Dyn-HaMR’s fork: https://github.com/ZhengdiYu/hamer



## Installation


```bash
conda create -n dynhamr python=3.12 -y

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install git+https://github.com/facebookresearch/detectron2 --no-build-isolation

pip install -e .
```