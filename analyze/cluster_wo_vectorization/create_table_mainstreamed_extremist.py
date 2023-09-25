import pandas as pd
import shared_functions as sf
import json
def import_data(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def do_clustering(hvv, data, n_clusters):
    indices = [x['_id']["$oid"] for x in data if 'embedding_result' in x.keys() for elt in x['embedding_result'][hvv]]
    embeddings = [elt for x in data if 'embedding_result' in x.keys() for elt in x['embedding_result'][hvv]]
    text = [elt for x in data if 'embedding_result' in x.keys() for elt in x['denoising_result'][hvv]]

    filename = f'agglom_clusering_{hvv}_{n_clusters}.pkl'

    clustering = sf.import_pkl_file(filename)

    labels = list(clustering.labels_)
    clusters = {c: [] for c in range(n_clusters)}
    for i in range(len(indices)):
        data_point = [x for x in data if x['_id']["$oid"] == indices[i]][0]
        # also add text
        clusters[labels[i]].append([indices[i], data_point['sample_id'], text[i]])
    return clusters

N_CLUSTERS = 5000
HVV = 'hero'
data = import_data('../../input/initial_subsample_results.json')
cluster_df = pd.read_csv(f'cluster_interpretation_{HVV}_{N_CLUSTERS}.csv')
main_extreme_df = cluster_df.loc[(cluster_df['extreme']>=-0.1) & (cluster_df['mainstream']>=-0.1) & (cluster_df['time']<=0)]
main_extreme_ids = list(main_extreme_df['cluster_id'].values)
clusters = do_clustering(HVV,data,N_CLUSTERS)
main_extreme_clusters= {k:clusters[k] for k in main_extreme_ids}
print('')