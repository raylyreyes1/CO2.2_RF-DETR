CO2.2 Final Project (RF-DETR for Object Detection)
========
RFE Module implementation to Facebook Research DETR model.

Facebook Research DETR Repository: https://github.com/facebookresearch/detr/tree/main

Facebook Research's Jupyter Notebook guide:

We provide a few notebooks in colab to help you get a grasp on DETR:
* [DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures of the paper)
* [Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): In this notebook, we demonstrate how to implement a simplified version of DETR from the grounds up in 50 lines of Python, then visualize the predictions. It is a good starting point if you want to gain better understanding the architecture and poke around before diving in the codebase.
* [Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): Demonstrates how to use DETR for panoptic segmentation and plot the predictions.


# Usage - Object detection

First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
The directory should be:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training and Evaluation
We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
A single epoch takes 28 minutes, so 300 epoch training
takes around 6 days on a single machine with 8 V100 cards.

To Train Single Node with ResNet50:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --lr_drop 400 --epochs 500 \
    --coco_path /path/to/coco
```
Evaluation line:
```
python main.py --batch_size 2 --no_aux_loss --eval \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /path/to/coco
```

To Train Multiple Node with ResNet50:
```
python run_with_submitit.py \
    --nodes 8 --timeout 3200 \
    --batch_size 1 --dilation \
    --lr_drop 400 --epochs 500 \
    --coco_path /path/to/coco
```
Evaluation line:
```
python main.py --no_aux_loss --eval \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
    --coco_path /path/to/coco
```


To Train Single Node with ResNet101:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --backbone resnet101 \
    --lr_drop 400 --epochs 500 \
    --coco_path /path/to/coco
```
Evaluation line:
```
python main.py --batch_size 2 --no_aux_loss --eval \
    --backbone resnet101 \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth \
    --coco_path /path/to/coco
```


To Train Multiple Node with ResNet101:
```
python run_with_submitit.py \
    --nodes 8 --timeout 3200 \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --lr_drop 400 --epochs 500 \
    --coco_path /path/to/coco
```
Evaluation line:
```
python main.py --no_aux_loss --eval \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /path/to/coco
```


To Evaluate Panoptic with ResNet50:
```
python main.py \
    --dilation \
    --batch_size 1 --no_aux_loss --eval \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth \
    --masks --dataset_file coco_panoptic \
    --coco_path /path/to/coco/ \
    --coco_panoptic_path /path/to/coco_panoptic
```


To Evaluate Panoptic with ResNet101:
```
python main.py \
    --backbone resnet101 \
    --batch_size 1 --no_aux_loss --eval \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth \
    --masks --dataset_file coco_panoptic \
    --coco_path /path/to/coco/ \
    --coco_panoptic_path /path/to/coco_panoptic
```


To Evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```


## Multinode training
Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```


Train baseline DETR-6-6 model on 4 nodes for 300 epochs:
```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```


# License
DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
