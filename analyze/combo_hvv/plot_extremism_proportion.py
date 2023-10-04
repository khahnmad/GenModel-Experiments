import pandas as pd
import shared_functions as sf
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_cluster_df(hvv_temp, n_clusters, single_combo, vers):
    if vers ==0:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
        else:
            cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
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


def fetch_y(N_CLUSTERS, template, vers):
    # N_CLUSTERS =
    SINGLE_COMBO = 'combo'

    cluster_df = load_cluster_df(hvv_temp=template, n_clusters=N_CLUSTERS, single_combo=SINGLE_COMBO, vers=vers)

    main_extreme = len(cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['mainstream']>=-0.5)])
    only_extreme = len(cluster_df.loc[(cluster_df['extreme']>=np.inf)])
    only_main = len(cluster_df.loc[(cluster_df['mainstream']>=np.inf)])
    other = len(cluster_df) - main_extreme - only_extreme - only_main
    return [main_extreme, only_extreme, only_main, other]

# n_clusters = 3500
# temp = 'b'

n_clusters= 2500
temp = 'c'
vers =1

# create data
x = ['HVV']
y = fetch_y(n_clusters,temp, vers)

y4 = np.array([y[0]])
y3 = np.array([y[1]])
y2 = np.array([y[2]])
y1 = np.array([y[3]])

# plot bars in stack manner
plt.bar(x, y1, color='g')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1 + y2, color='y')
plt.bar(x, y4, bottom=y1+y2+y3, color='r')
plt.xlabel("HVV Combination")
plt.ylabel("Number of Clusters")
plt.legend([ "Remaining Clusters","Only Mainstream","Only Extremist","Mainstreamed Extremist Candidates"])
plt.title("Partisanship Proportion of Clusters")
plt.show()