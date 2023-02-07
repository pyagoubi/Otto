# Otto

Description [Link](https://www.kaggle.com/competitions/otto-recommender-system)

## Objective  

The goal is to predict e-commerce clicks, cart additions, and orders with a recommender system based on previous events in a user session. The transformer model predicts the next product IDs, that the customer interacts with.

## Files 
* Dataset **[Link](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint)**
* Data Preprocessing **[Link](https://github.com/pyagoubi/Otto/blob/main/preprocessing.py)**
* BERT4Rec Model **[Link](https://github.com/pyagoubi/Otto/blob/main/model.py)**
* Word2Vec Model (for product ID embeddings) **[Link](https://github.com/pyagoubi/Otto/blob/main/w2vec.py)**
* Training **[Link](https://github.com/pyagoubi/Otto/blob/main/training.py)**
* Background: **[Link](https://arxiv.org/abs/1904.06690)** 

## Run script
```
python training.py -i <dataset path> -o <output directory> --epochs <No of epochs: int>
```
## Notebooks
* Preprocessing **[Link](https://github.com/pyagoubi/Otto/blob/main/Notebooks/otto-prep-training-and-validation-sets.ipynb)**
* Model + Training **[Link](https://github.com/pyagoubi/Otto/blob/main/Notebooks/bert4rec.ipynb)**
