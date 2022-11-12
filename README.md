# ViSeH
## Overview
This project includes source codes and examples of ViSeH. 

For quick start of using ViSeH, we provide runnable testing examples, including [pretrained models](./model_save/resnet18), a [sub-hierarchy](./hierarchy), and several [samples](./data_food101_demo) from the Ingredient-101 tesing set.
## Environment
* Pytorch >= 1.10.2
* Python >= 3.6.5
* torchvision >= 0.2.1
## Examples
We provide some examples to demonstrate the mechanism of ViSeH.

    python test.py --result_path result/ --hierarchy_path hierarchy/ --art_epoch 2

Here is one of the cases: classifying an image with ground-truth label **1** by the global model (Visual backbone) outputs the **wrong Top-1 prediction**, while the correct prediction is at **Top-4**. By refining of the VSRF and CAGL modules, the model finally outputs the **correct Top-1 prediction**.

> Sample 0, Ground Truth label: 1

> Top-1 Class prediction: Global model: [20], VSRF: [1], CAGL: [1] | final prediction: [1]

> Top-2 Class prediction: Global model: [77], VSRF: [1], CAGL: [50]

> Top-3 Class prediction: Global model: [38], VSRF: [77], CAGL: [20]

> Top-4 Class prediction: Global model: [1], VSRF: [1], CAGL: [23]

> Top-5 Class prediction: Global model: [42], VSRF: [1], CAGL: [77]


## Training
We also provide complete training codes for reproduction. Please refer to [option](./opts.py) to see more parameters.
### Offline training
####
**Obtaining FVSA model** for matching visual-semantic pairs and **creating Hierarchy**.

    python train_offline.py --result_path result/ --hierarchy_path [PATH_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
### Online training
Training **CAGL model** and **Fusion model**.

    python train_oneline.py --result_path result/ --hierarchy_path [PATH_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
## Testing
    python test.py --hierarchy_path [PATH_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
