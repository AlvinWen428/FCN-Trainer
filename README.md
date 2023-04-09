# FCN-Trainer

PyTorch implementation of FCN. This codebase is based on [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet).

## Installation

1. Python 3 and pip
2. Set up a virtual environment (optional, but recommended)
3. Install dependencies using pip: `pip install -r requirements.txt`

## Train FCN on Cityscapes

```
python main.py -m train --save-dir save/folder/ --name FCN --model-name fcn --dataset cityscapes --dataset-dir path/root_directory/ -lr 7e-4 --batch-size 16 --height 256 --width 512 --regression
```
