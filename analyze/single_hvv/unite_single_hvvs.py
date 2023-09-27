import pandas as pd
import shared_functions as sf

def load_cluster_df(hvv_temp, n_clusters):
    cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    return cluster_df

def load_raw_data():
    return sf.import_json('../../input/initial_subsample_results.json')

def fetch_mainstream_extreme_clusters(hvv, n_clusters):
    cluster_df = load_cluster_df(hvv, n_clusters)
    main_extreme_df = cluster_df.loc[
        (cluster_df['extreme'] >= -0.5) & (cluster_df['mainstream'] >= -0.5) & (cluster_df['time'] <= 0)]
    main_extreme_ids = list(main_extreme_df['cluster_id'].values)
    return main_extreme_ids


def fetch_cluster_components(raw_data , ids, hvv):


    clustering = sf.import_pkl_file( f'agglom_clusering_{hvv}_{n_clusters}.pkl')
    labels = list(clustering.labels_)

    hvv_data = []
    for i in range(len(raw_data)):
        elt = raw_data[i]
        if 'embedding_result' not in elt.keys():
            continue
        for j in range(len(elt['embedding_result'][hvv])):
            new_elt = {'_id': elt['_id']["$oid"],
                       'sample_id':elt['sample_id'],
                       'text':elt['denoising_result'][hvv][j],
                       'cluster_id': int(labels[i])}
            hvv_data.append(new_elt)

    clusters = {c: [] for c in ids}
    for i in range(len(hvv_data)):
        cluster_id = hvv_data[i]['cluster_id']
        if cluster_id in clusters.keys():
            clusters[cluster_id].append(hvv_data[i])

    return clusters

## Get mainstream extremes for each of the three hvvs ##
n_clusters = 5000
data = load_raw_data()

cluster_components = {k:[] for k in ['hero', 'villain', 'victim']}
for hvv in ['hero','villain','victim']:
    cluster_ids = fetch_mainstream_extreme_clusters( hvv, n_clusters)
    cluster_components[hvv] = fetch_cluster_components(data, cluster_ids,hvv)

h_ids = [item["_id"] for k in cluster_components['hero'].keys() for item in cluster_components['hero'][k]]
vil_ids = [item["_id"] for k in cluster_components['villain'].keys() for item in cluster_components['villain'][k]]
vic_ids = [item["_id"] for k in cluster_components['victim'].keys() for item in cluster_components['victim'][k]]

shared = []
for h in h_ids:
    if h in vic_ids and h in vil_ids:
        shared.append(h)

shared_clusters = []
for _id in shared:
    h_clusters = {k:cluster_components['hero'][k] for k in cluster_components['hero'].keys()
                  if _id in [x["_id"] for x in  cluster_components['hero'][k] ]}
    vil_clusters = {k:cluster_components['villain'][k] for k in cluster_components['villain'].keys()
                  if _id in [x["_id"] for x in  cluster_components['villain'][k] ]}
    vic_clusters = {k: cluster_components['victim'][k] for k in cluster_components['victim'].keys()
                    if _id in [x["_id"] for x in cluster_components['victim'][k]]}
    print('')




# for s in shared:






