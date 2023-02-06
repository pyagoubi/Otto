import pandas as pd
import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def train_w2v(input_file, output_path):

    train = pd.read_parquet(input_file)

    aid_sorted = sorted(list(train['aid'].unique()))
    mapping = {k: i + 2 for i, k in enumerate(aid_sorted)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    train['aid'] = train['aid'].map(mapping)

    train.loc[len(train.index)]  = [19000000, 0,0,0]

    train.session = train.session.astype('int32')
    train.ts = train.ts.astype('int32')
    train.type =train.type.astype('uint8')

    train = pl.DataFrame(train)

    sentences_df = train.groupby('session').agg(
    pl.col('aid').alias('sentence')
    )

    sentences = sentences_df['sentence'].to_list()

    w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)
    w2vec.save(output_path+'\word2vec.model')
