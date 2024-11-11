# BEV-CV - (Processing codebase ASAP)

This repository is the official implementation of [BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation](https://arxiv.org/abs/2312.15363), IROS 2024. 

![network](https://github.com/user-attachments/assets/86c96ec2-c599-4bac-8d58-01d6fb30efe5)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Datasets - CVUSA & CVACT
We use two existing dataset to do the experiments

  CVUSA: sampled across the US, ground-level panoramas and corresponding satellite images.
         The dataset can be accessed from https://github.com/viibridges/crossnet

  CVACT: sampled across Australia, ground-level panoramas and corresponding satellite images.
         The dataset can be accessed from https://github.com/Liumouliu/OriCNN

#### Data Preparation
We use limited-FOV panorama crops - emulating a vehicle mounted front-facing monocular camera. To achieve this - the data can be preprocessed before evaluation by running the script:
```
python run.....
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

## Results

Our model achieves the following performance on :

### Image Retrieval on CVUSA & CVACT

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |


## Citation
This work is published in IROS 2024.
If you are interested in our work and/or use our code, please include the following citation in your work:

```
  @INPROCEEDINGS{bevcv,
    author={Shore, Tavis and Hadfield, Simon and Mendez, Oscar },
    booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
    title={BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation}, 
    year={2024},
    pages={11047-11054},
  }
```
## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
