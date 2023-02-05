import gc
import os
import warnings
import numpy as np
import pandas as pd
import glob

from tqdm import tqdm
from datetime import datetime, timedelta, time


def get_train_target(input_files, output_directory):

    train = pd.read_parquet(input_files)
    train['ts'] = pd.to_datetime(train['ts']*1e9)
    
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
                train_df.to_parquet(f'{output_directory}/train_w{k}_part{j}.parquet')
                label_df.to_parquet(f'{output_directory}/label_w{k}_part{j}.parquet')
                j += 1
                del train_list, label_list, label_df, train_df
                train_list = []
                label_list = []          
                gc.collect()
        del train_temp
        gc.collect()


def read_and_concatenate_parquet_files(pattern):
    """Read and concatenate parquet files matching given pattern."""
    parquet_files = {}
    for file in glob.glob(pattern):
        parquet_files[file] = pd.read_parquet(file)
    return pd.concat(parquet_files.values())

def create_column_mapping(df, col_name):
    # Read in the data from the parquet file
    all_aid = pd.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/train.parquet')
    
    # Get a sorted list of the unique values in the specified column
    aid_sorted = sorted(list(all_aid[col_name].unique()))
    
    # Delete the dataframe to save memory
    del all_aid
    
    # Create a mapping from the unique values to integers, starting from 2
    mapping = {k: i + 2 for i, k in enumerate(aid_sorted)}
    
    # Create an inverse mapping from the integers back to the unique values
    inverse_mapping = {v: k for k, v in mapping.items()}
    
    return mapping, inverse_mapping

def map_column(df, col_name, mapping):
    # Replace the values in the specified column with their corresponding integer values
    df[col_name] = df[col_name].map(mapping)
    return df

def get_merged_sessions(session_df, label_session_df):
    """
    Merge the session data and label data by session.

    """
    input_df = pd.DataFrame(session_df.groupby('session')['aid'].unique().agg(list))
    label_df = pd.DataFrame(label_session_df.groupby('session')['aid'].unique().agg(list))
    
    # Rename the 'aid' column to 'input'
    input_df = input_df.rename(columns={'aid': 'input'})
    
    # Rename the 'aid' column to 'label'
    label_df = label_df.rename(columns={'aid': 'label'})

    # Merge the input and label dataframes on the 'session' column, using 'session' as the index
    merged_df = input_df.merge(label_df, left_index=True, right_index=True)
    
    return merged_df