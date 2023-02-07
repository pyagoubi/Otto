import pandas as pd
import pytorch_lightning as pl
import torch
import gensim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model import Recommender
from w2vec import train_w2v
from preprocessing import get_train_target,read_and_concatenate_parquet_files, create_column_mapping, map_column, get_merged_sessions


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sessions, input_length, target_length):
        self.sessions = sessions
        self.input_length = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.sessions)
    
    def pad_items(self, session, length, input_item = True):
        if len(session)< length:
            session = session + list((length - len(session)) * [0])
        else: 
            if input_item:
                session = session[-length:]
            else: session = session[:length]
        return session



    def __getitem__(self, idx):
        input_tokens = self.sessions.iloc[idx, self.sessions.columns.get_loc("input")]
        
        length_input = len(input_tokens)
        input_tokens = self.pad_items(input_tokens, self.input_length)
            
        target = self.sessions.iloc[idx, self.sessions.columns.get_loc("label")]
        target = self.pad_items(target, self.target_length, input_item = False)
        
        target_mask = [1 if item != 0 else item for item in target ]
        
        
        target = input_tokens +target
        input_tokens = input_tokens + target_mask 
        
 
        
        input_tokens_orig = torch.tensor(input_tokens)
        input_tokens = w2v.wv[input_tokens]
        
        input_tokens = torch.tensor(input_tokens, dtype = torch.float )
        target = torch.tensor(target, dtype = torch.long )

        
        
        mask = torch.eq(input_tokens_orig, 0).type(torch.bool).unsqueeze(0).permute(1,0)
        
       
        return input_tokens, target, mask, input_tokens_orig


def train(
    output_path: str,
    input_file: str,
    log_dir: str = "recommender_logs",
    model_dir: str = "recommender_models",
    batch_size: int = 10,
    epochs: int = 20,
    embedding_length = 32,
    target_length = 20,
    sequence_length = 100):


    get_train_target(input_file, output_path)

    # Read and concatenate training data from first week
    train_w1 = read_and_concatenate_parquet_files(f"{output_path}/train_w0_part*.parquet")
    label_w1 = read_and_concatenate_parquet_files(f"{output_path}/label_w0_part*.parquet")

    # Read and concatenate training data from fourth week for validation
    train_w4 = read_and_concatenate_parquet_files(f"{output_path}/train_w3_part*.parquet")
    label_w4 = read_and_concatenate_parquet_files(f"{output_path}/label_w3_part*.parquet")

    # Create the mapping and inverse mapping for the 'aid' column
    mapping, inverse_mapping = create_column_mapping(train_w1, 'aid', input_file)

    print(len(mapping))

    # Replace the values in the 'aid' column of the dataframe with their corresponding integer values
    train_w1 = map_column(train_w1, 'aid', mapping)
    train_w4 = map_column(train_w4, 'aid', mapping)
    #discard sessions with NaN aid (in 4th week but not in first)
    train_w4 = train_w4[~train_w4.session.isin(train_w4[train_w4['aid'].isna()]['session'].unique())]
    
    train_w1 = get_merged_sessions(train_w1, label_w1)
    train_w4 = get_merged_sessions(train_w4, label_w4)

    # train Word2vec
    train_w2v(input_file, output_path)
    w2v = gensim.models.Word2Vec.load(f"{output_path}\word2vec.model")


    train_data = Dataset(
    sessions=train_w1,
    input_length = sequence_length,
    target_length=target_length
)

    val_data = Dataset(
    sessions = train_w4,
    input_length = sequence_length,
    target_length=target_length
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))


    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
    )

    model = Recommender(
        out = len(mapping)+2,
        channels=embedding_length,
        dropout=0.2,
        lr=1e-4,
        word2vec = w2v
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1
    )

    logger = TensorBoardLogger(
        save_dir='/kaggle/working/',
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath='/kaggle/working/',
        filename="recommender",
        )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file")
    parser.add_argument("-o","--output_path")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train(
        output_path=args.output_path,
        input_file=args.input_file,
        epochs=args.epochs,
    )
