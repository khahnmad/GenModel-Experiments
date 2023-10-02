import pandas as pd
import shared_functions as sf
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_cluster_df(hvv_temp, n_clusters, single_combo, vers=0):
    if vers==0:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
        else:
            cluster_df = pd.read_csv(f'combo_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    else:
        cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
    return cluster_df


def import_raw_data(hvv_temp):
    if hvv_temp != 'a' and hvv_temp!='b':
        with open('../input/initial_subsample_results.json', 'r') as j:
            content = json.loads(j.read())
    else:
        with open(f'../clustering/cluster_experiments/sbert_embdddings/initial_subsample_{hvv_temp}.pkl', "rb") as f:
            content = pickle.load(f)['content']
            f.close()
    return content


def do_clustering(hvv, data, n_clusters, single_combo):


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

    clustering = sf.import_pkl_file(filename)

    labels = list(clustering.labels_)
    clusters = {c: [] for c in range(n_clusters)}
    for i in range(len(indices)):
        data_point = [x for x in data if x['_id']["$oid"] == indices[i]][0]
        # also add text
        clusters[labels[i]].append([indices[i], data_point['sample_id'], text[i]])
    return clusters


def fetch_y(hvv, n_clusters, vers=0):


    cluster_df = load_cluster_df(hvv, n_clusters=n_clusters, single_combo='single',vers=vers)

    main_extreme = len(cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['mainstream']>=-0.5)])
    only_extreme = len(cluster_df.loc[(cluster_df['extreme']>=np.inf)])
    only_main = len(cluster_df.loc[(cluster_df['mainstream']>=np.inf)])
    other = len(cluster_df) - main_extreme - only_extreme - only_main
    return [main_extreme, only_extreme, only_main, other]

N_CLUSTERS = 2500
VERS = 1
# create data
x = ['Hero', 'Villain', 'Victim']
h = fetch_y('hero',N_CLUSTERS,VERS)
vil = fetch_y('villain',N_CLUSTERS, VERS)
vic = fetch_y('victim',N_CLUSTERS, VERS)

y4 = np.array([h[0], vil[0],vic[0]])
y3 = np.array([h[1], vil[1],vic[1]])
y2 = np.array([h[2], vil[2],vic[2]])
y1 = np.array([h[3], vil[3],vic[3]])

# plot bars in stack manner
plt.bar(x, y1, color='g')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1 + y2, color='y')
plt.bar(x, y4, bottom=y1+y2+y3, color='r')
plt.xlabel("Archetypes")
plt.ylabel("Number of Clusters")
plt.legend([ "Remaining Clusters","Only Mainstream","Only Extremist","Mainstreamed Extremist Candidates"])
plt.title("Partisanship Proportion of Clusters")
plt.show()