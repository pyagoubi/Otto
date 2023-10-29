# Otto

Description [Link](https://www.kaggle.com/competitions/otto-recommender-system)

## Objective  

The goal is to predict e-commerce clicks, cart additions, and orders with a recommender system based on previous events in a user session. The transformer model predicts the next product IDs, that the customer interacts with.

## Files 
* Dataset **[Link](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint)**
* Background: **[Link](https://arxiv.org/abs/1904.06690)** 

## Instructions
1. Clone the repo
2. Download the dataset
3. Use requirements.txt to install required libraries
4. Run
```
python training.py -i <path to train.parquet> -o <output directory> --epochs <No of epochs: int>
```
## Notebooks
* Preprocessing **[Link](https://github.com/pyagoubi/Otto/blob/main/Notebooks/otto-prep-training-and-validation-sets.ipynb)**
* Model + Training **[Link](https://github.com/pyagoubi/Otto/blob/main/Notebooks/bert4rec.ipynb)**
