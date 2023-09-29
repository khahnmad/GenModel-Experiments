import pandas as pd
import shared_functions as sf
import json
import pickle
import numpy as np

def load_cluster_df(n=5):
    if n == 5:
        data = sf.import_json('alphabet_clusters.json')['content']
        df = pd.DataFrame(data=data[1:], columns=data[0])
    else:
        data = sf.import_json(f'alphabet_clusters_{n}.json')['content']
        df = pd.DataFrame(data=data[1:], columns=data[0])
    return df



cluster_df = load_cluster_df(5)

main_extreme_df = cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['main']>=-0.5)]
pos_slope = len(main_extreme_df.loc[main_extreme_df['slope']>0])
null_slope = len(main_extreme_df.loc[main_extreme_df['slope'] == 0])
neg_slope = len(main_extreme_df.loc[main_extreme_df['slope'] < 0])
print('')
