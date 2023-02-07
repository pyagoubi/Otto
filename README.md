# Otto

Description [Link](https://www.kaggle.com/competitions/otto-recommender-system)

## Objective  

The goal is to predict e-commerce clicks, cart additions, and orders with a multi-objective recommender system based on previous events in a user session.

<b>Links  </b>
* Dataset **[Link](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint)**
* Data Preprocessing **[Link](https://github.com/pyagoubi/Otto/blob/main/preprocessing.py)**
* BERT4Rec Model **[Link](https://github.com/pyagoubi/Otto/blob/main/model.py)**
* Training **[Link](https://github.com/pyagoubi/Otto/blob/main/training.py)**
* Background: **[Link](https://arxiv.org/abs/1904.06690)** 

Run script
```
python training.py -i <dataset path> -o <output directory> --epochs <No of epochs: int>
```
