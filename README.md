# ViSeH
## Overview
This project includes source codes and examples of ViSeH. 

For quick start of using ViSeH, we provide runnable testing examples, including [pretrained models](./model_save/resnet18), a [sub-hierarchy](./hierarchy), and several [samples](./data_food101_demo) from the Ingredient-101 tesing set.
## Environment
* Pytorch >= 1.10.2
* Python >= 3.6.5
* torchvision >= 0.2.1
## Demo
We provide some samples

    python test.py --result_path result/ --path_stage1 hierarchy/ --art_epoch 2

Here is one of the cases: classifying an image with ground-truth label **1** by the global model (Visual backbone) outputs **wrong Top-1 prediction**, while the correct predition is at **Top-4**. By refining of the VSHC and MMGF modules, the model finally outputs **correct Top-1 prediction**.

> Sample 0, Ground Truth label: 1

> Top-1 Class prediction: Global model: [20], VSHC: [1], MMGF: [1] | final prediction: [1]

> Top-2 Class prediction: Global model: [77], VSHC: [1], MMGF: [50]

> Top-3 Class prediction: Global model: [38], VSHC: [77], MMGF: [20]

> Top-4 Class prediction: Global model: [1], VSHC: [1], MMGF: [23]

> Top-5 Class prediction: Global model: [42], VSHC: [1], MMGF: [77]


## Training
We also provide complete training code.
### Offline training
####
**Obtaining FVSA model** for matching visual-semantic pairs and **creating Hierarchy**.

    python train_offline.py --result_path result/ --path_stage1 result/ --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
### Online training
Training **MMGF model** and **Fusion model**.

    python train_oneline.py --result_path result/ --path_stage1 result/ --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
## Testing
    python test.py --path_stage1 [PATH_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
