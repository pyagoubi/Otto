import gc
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from datetime import datetime, timedelta, time

%%time

train = pd.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/train.parquet')
train['ts'] = pd.to_datetime(train['ts']*1e9)

%%time

def get_train_target():
    
    
    delta = timedelta(days = 6, hours = 23, minutes = 59, seconds = 59)
    
    for k in range(4):
        
        if k == 0: train_temp = train[(train.ts >= train.ts.min()) & (train.ts <= train.ts.min() + (k+1)* delta)].reset_index(drop = True)
        else: train_temp = train[(train.ts > train.ts.min() + k* delta) & (train.ts <= train.ts.min() + (k+1)* delta)].reset_index(drop = True)
        
        # #drop one_event_sessions
        one_event_df = train_temp.groupby('session', as_index = False).count()
        one_event_df = one_event_df[one_event_df['aid']==1]
        one_sessions =  one_event_df['session']
        train_temp = train_temp[~train_temp['session'].isin(one_sessions)]
        del one_event_df, one_sessions


        train_list = []
        label_list = []
        j=0
        end = len(train_temp.session.unique())


        for i, grp in tqdm(enumerate(train_temp.groupby('session'))):
            cutoff = np.random.randint(1, grp[1].shape[0]) 
            train_list.append(grp[1].iloc[:cutoff])
            label_list.append(grp[1].iloc[cutoff:])
            if (i % 200000 == 0) or (i == end - 1):
                train_df = pd.concat(train_list).reset_index(drop=True)
                label_df = pd.concat(label_list).reset_index(drop=True)
                train_df.to_parquet(f'/kaggle/working/train_w{k}_part{j}.parquet')
                label_df.to_parquet(f'/kaggle/working/label_w{k}_part{j}.parquet')
                j += 1
                del train_list, label_list, label_df, train_df
                train_list = []
                label_list = []          
                gc.collect()
        del train_temp
        gc.collect()


get_train_target()