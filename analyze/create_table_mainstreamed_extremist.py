import pandas as pd
import shared_functions as sf
import json
import pickle
import numpy as np

def load_cluster_df(hvv_temp, n_clusters, single_combo, vers=0):
    if vers ==0:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'single_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
        else:
            cluster_df = pd.read_csv(f'combo_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    else:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'single_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
        else:
            cluster_df = pd.read_csv(f'combo_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
    return cluster_df


def import_raw_data(hvv_temp, vers=0):
    if vers == 0:
        if hvv_temp != 'a' and hvv_temp!='b':
            with open('../input/initial_subsample_results.json', 'r') as j:
                content = json.loads(j.read())
        else:
            with open(f'../clustering/combined_hvv/sbert_embdddings/initial_subsample_{hvv_temp}.pkl', "rb") as f:
                content = pickle.load(f)['content']
                f.close()
    else:
        if hvv_temp != 'a' and hvv_temp != 'b' and hvv_temp!='c':
            with open('../input/initial_subsample_triplets_results.json', 'r') as j:
                content = json.loads(j.read())
        else:
            with open(f'../clustering/initial_subsample_{hvv_temp}_v{vers}.pkl',
                      "rb") as f:
                content = pickle.load(f)['content']
                f.close()


    return content


def do_clustering(hvv, data, n_clusters, single_combo,vers):


    if single_combo=='single':
        indices = [x['_id']["$oid"] for x in data if 'embedding_result' in x.keys() for elt in
                   x['embedding_result'][hvv]]
        embeddings = [elt for x in data if 'embedding_result' in x.keys() for elt in x['embedding_result'][hvv]]
        text = [elt for x in data if 'embedding_result' in x.keys() for elt in x['denoising_result'][hvv]]
        filename = f'single_hvv/agglom_clusering_{hvv}_{n_clusters}.pkl'
    else:
        indices = [x['_id']["$oid"] for x in data if 'embedding' in x.keys()]
        embeddings = [x['embedding'] for x in data if 'embedding' in x.keys()]
        text = [x['sentence'] for x in data if 'embedding' in x.keys()]
        filename = f'combo_hvv/agglom_clusering_{hvv}_{n_clusters}.pkl'
    if vers!=0:
        if single_combo=='single': # TODO THIS is still messed up
            indices = [x['_id']["$oid"] for x in data if 'embedding_result' in x.keys()]
            embeddings = [x['embedding'] for x in data if 'embedding_result' in x.keys()]
            text = [x['sentence'] for x in data if 'embedding_result' in x.keys()]
            filename = f'single_hvv/agglom_clusering_{hvv}_{n_clusters}_v{vers}.pkl'
        else:
            indices = [x['_id']["$oid"] for x in data if 'embedding' in x.keys()]
            embeddings = [x['embedding'] for x in data if 'embedding' in x.keys()]
            text = [x['sentence'] for x in data if 'embedding' in x.keys()]
            filename = f'combo_hvv/agglom_clusering_{hvv}_{n_clusters}_v{vers}.pkl'

    clustering = sf.import_pkl_file(filename)

    labels = list(clustering.labels_)
    clusters = {c: [] for c in range(n_clusters)}
    for i in range(len(indices)): # TODO : somethings gone wrong here
        data_point = [x for x in data if x['_id']["$oid"] == indices[i]][0]
        # also add text
        if vers==0:
            clusters[labels[i]].append([indices[i], data_point['sample_id'], text[i]])
        else:
            clusters[labels[i]].append([indices[i], data_point['publish_date'], data_point['partisanship'], text[i]])
    return clusters

# N_CLUSTERS = 3500
# HVV_TEMP = 'b'
# SINGLE_COMBO = 'combo'

# N_CLUSTERS = 3500
# HVV_TEMP = 'b'
# SINGLE_COMBO = 'combo'

# N_CLUSTERS = 2500
# HVV_TEMP = 'villain'
# SINGLE_COMBO = 'single'
# VERS = 1

N_CLUSTERS = 2500
HVV_TEMP = 'c'
SINGLE_COMBO = 'combo'
VERS = 1

data = import_raw_data(HVV_TEMP, vers=VERS)

cluster_df = load_cluster_df(HVV_TEMP, n_clusters=N_CLUSTERS, single_combo=SINGLE_COMBO, vers=VERS)

main_extreme_df = cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['mainstream']>=-0.5) & (cluster_df['time']<0)]
main_extreme_ids = list(main_extreme_df['cluster_id'].values)
clusters = do_clustering(HVV_TEMP,data,N_CLUSTERS,single_combo=SINGLE_COMBO,vers=VERS)
main_extreme_clusters= {k:clusters[k] for k in main_extreme_ids}
# print('')

only_extreme_df = cluster_df.loc[(cluster_df['extreme']>=np.inf)]
extreme_ids = list(only_extreme_df['cluster_id'].values)
extreme_clusters= {k:clusters[k] for k in extreme_ids}
print('')

only_main_df = cluster_df.loc[(cluster_df['mainstream']>=np.inf)]
mainstream_ids = list(only_main_df['cluster_id'].values)
mainstream_clusters= {k:clusters[k] for k in mainstream_ids}
print('')

# Strongest slope
# strongslope_df = cluster_df.sort_values(by='time',ascending=False)
strongslope_df= cluster_df.loc[(cluster_df['time']>=1)]
strongslope_ids = list(strongslope_df['cluster_id'].values)
strongslope_clusters= {k:clusters[k] for k in strongslope_ids}
print('')