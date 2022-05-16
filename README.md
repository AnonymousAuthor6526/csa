

# Learning to Regress Occluded Human Body with Pseudo Surface Representation

This repository is the official implementation of [Learning to Regress Occluded Human Body with Pseudo Surface Representation](https://arxiv.org/abs/2030.12345). 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training

To train the model(s) in the paper, run this command:

```train
python train.py --cfg configs/vit_train.yaml
```



## Evaluation

To evaluate my model on the benchmark of 3DPW and OCHuman, go to the configs/vit_test.yaml ,set the RUN_TEST = True, run:

```eval
python train.py --cfg configs/vit_test.yaml
```




## Pre-trained Models

You can download pretrained models here:

- [MPQA Pre-trained Models](https://drive.google.com/file/d/1WojbZvLfGFS8OzcRplwPIw2EeRWiGNd_/view?usp=sharing) trained on 3DPW and OCHuman. 



## Results

Our model achieves the following performance on :

### [3D  on the benchmark of 3DPW and OCHuman]

You should obtain results in this table on 3DPW test set:

| Method.            |    MPJPE↓       |    PA-MPJPE↓   |      PVE↓     |
| ------------------ |---------------- | -------------- | ------------- |
| MPQA w. 3DPW       |     71.6        |      44.8      |      83.4     |



## Contributing

