

# Learning to Regress Occluded Human Body with Pseudo Surface Representation

![framework](/assets/Fig1.png)

This implementation includes evaluation and training code for SSPA and MPQA implemented in PyTorch. SSPA is a semi-supervised pseudo-attention framework for 3D human body estimation, learning from unlabeled single view. MPQA is a lightweight and occlusion-robust human pose and shape regressor.

## Getting Started

The method has been implemented and tested on Ubuntu 20.04 with Python 3.7.

```Clone the repo:
https://github.com/AnonymousAuthor6526/mpqa.git
```

Install the requirements using conda:
```shell
# conda
source install_conda.sh
```

Preparation of Data:
```shell
source prepare_data.sh
```


```shell
data/
├── dataset_extras # Contains all pre-processed data
├── dataset_folders # Contains all datasets
├── sspa_w_3dpw_checkpoint.ckpt  # checkpoint with 3DPW training
├── mpqa_wo_3dpw_checkpoint.ckpt # checkpoint without 3DPW training
├── smpl 
│   └── ... # SMPL Models
└── ...
```

```shell
dsr_data/
├── dataset_extras # Contains all pre-processed data
├── dataset_folders # Contains all datasets
│   ├── coco
│   │   ├── grph_sequences # Contains psuedo-GT Graphonomy Segmentation
│   │   └── ...
│   └── h36m
│   │   ├── grph_sequences # Contains psuedo-GT Graphonomy Segmentation
│   │   └── ...
│   └── ... # Other datasets
├── dsr_w3DPW_checkpoint.pt  # dsr_checkpoint with 3DPW training
├── dsr_wo3DPW_checkpoint.pt # dsr_checkpoint without 3DPW training
├── semantic_prior.npy # SMPL Semantic Prior
├── smpl 
│   └── ... # SMPL Models
└── ...
```

## Training

To train the model(s) in the paper, run this command:

```train
coming soon
```

## Evaluation

To evaluate the model on the benchmark of 3DPW:

```eval
python main.py --cfg configs/mpqa_test.yaml
```

## Pre-trained Models

You can download pretrained models here:

- [MPQA Pre-trained Models](https://drive.google.com/file/d/1WojbZvLfGFS8OzcRplwPIw2EeRWiGNd_/view?usp=sharing). 

## Results

Our model achieves the following performance on :

### [3D  on the benchmark of 3DPW]

You should obtain results in this table on 3DPW test set:

| Method.            |    MPJPE↓       |    PA-MPJPE↓   |      PVE↓     |
| ------------------ |---------------- | -------------- | ------------- |
| MPQA w. 3DPW       |     71.6        |      44.8      |      83.4     |


## License
This code is available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/AnonymousAuthor6526/mpqa/blob/main/LICENSE). By downloading and using this code you agree to the terms in the LICENSE. Third-party datasets and software are subject to their respective licenses.
