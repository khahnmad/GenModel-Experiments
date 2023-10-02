import pandas as pd
import shared_functions as sf

def load_cluster_df(hvv_temp, n_clusters, single_combo, vers=0):
    if vers==0:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
        else:
            cluster_df = pd.read_csv(f'combo_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    else:
        cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
    return cluster_df

def load_raw_data(vers=0):
    if vers==0:
        return sf.import_json('../../input/initial_subsample_results.json')
    else:
        return sf.import_json('../../input/initial_subsample_triplets_results.json')

def fetch_mainstream_extreme_clusters(hvv, n_clusters,vers):
    cluster_df = load_cluster_df(hvv, n_clusters,single_combo='single',vers=vers)
    main_extreme_df = cluster_df.loc[
        (cluster_df['extreme'] >= -0.5) & (cluster_df['mainstream'] >= -0.5) & (cluster_df['time'] <= 0)]
    main_extreme_ids = list(main_extreme_df['cluster_id'].values)
    return main_extreme_ids


def fetch_cluster_components(raw_data , ids, hvv, vers):
    indices = []
    embeddings = []
    text = []

    for i in range(len(raw_data)):
        elt = raw_data[i]
        if 'embedding_result' not in raw_data[i].keys():
            continue
        for j in range(len(raw_data[i]['embedding_result'][hvv])):
            indices.append(elt['_id']["$oid"])
            embeddings.append(elt['embedding_result'][hvv][j])
            text.append(elt['denoising_result'][hvv][j])

    if vers==0:
        clustering = sf.import_pkl_file( f'agglom_clusering_{hvv}_{n_clusters}.pkl')
    else:
        clustering = sf.import_pkl_file(f'agglom_clusering_{hvv}_{n_clusters}_v{vers}.pkl')
    labels = list(clustering.labels_)

    hvv_data = []
    # raw_data = [x for x in raw_data if 'embedding_result' in x.keys() and len(x['embedding_result'][hvv])>0]
    for i in range(len(indices)):
        elt = [x for x in raw_data if x['_id']['$oid']==indices[i]][0]
        if 'embedding_result' not in elt.keys():
            continue
        for j in range(len(elt['embedding_result'][hvv])):
            if vers==0:
                new_elt = {'_id': elt['_id']["$oid"],
                       'sample_id':elt['sample_id'],
                       'text':elt['denoising_result'][hvv][j],
                       'cluster_id': int(labels[i])}
            else:
                new_elt = {'_id': elt['_id']["$oid"],
                           'partisanship': elt['partisanship'],
                           'publish_date': elt['publish_date'],
                           'text': elt['denoising_result'][hvv][j],
                           'cluster_id': int(labels[i])}
            hvv_data.append(new_elt)

    clusters = {c: [] for c in ids}
    for i in range(len(hvv_data)):
        cluster_id = hvv_data[i]['cluster_id']
        if cluster_id in clusters.keys():
            clusters[cluster_id].append(hvv_data[i])

    return clusters

## Get mainstream extremes for each of the three hvvs ##
n_clusters = 2500
vers = 1
data = load_raw_data(vers)

cluster_components = {k:[] for k in ['hero', 'villain', 'victim']}
for hvv in ['hero','villain','victim']:
    cluster_ids = fetch_mainstream_extreme_clusters( hvv, n_clusters,vers)
    cluster_components[hvv] = fetch_cluster_components(data, cluster_ids,hvv,vers)

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






