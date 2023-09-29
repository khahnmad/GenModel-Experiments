import shared_functions as sf
import pandas as pd
import numpy as np

def load_cluster_df(n=5,vers=0):
    if vers==0:
        if n == 5:
            data = sf.import_json('alphabet_clusters.json')['content']
            df = pd.DataFrame(data=data[1:], columns=data[0])
        else:
            data = sf.import_json(f'alphabet_clusters_{n}.json')['content']
            df = pd.DataFrame(data=data[1:], columns=data[0])
    else:
        data = sf.import_json(f'alphabet_clusters_v{vers}_{n}.json')['content']
        df = pd.DataFrame(data=data[1:], columns=data[0])
    return df


cluster_df = load_cluster_df(2,1)
main_extreme_df = cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['main']>=-0.5) & (cluster_df['slope']<0)]

only_extreme_df = cluster_df.loc[(cluster_df['extreme']>=np.inf)]

only_main_df = cluster_df.loc[(cluster_df['main']>=np.inf)]
print('')
