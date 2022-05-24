

# Learning to Regress Occluded Human Body with Pseudo Surface Representation




## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training

To train the model(s) in the paper, run this command:

```train
coming soon
```



## Evaluation

To evaluate my model on the benchmark of 3DPW:

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

