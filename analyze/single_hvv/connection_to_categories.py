import pandas as pd
import shared_functions as sf

def load_cluster_df(hvv_temp, n_clusters, vers):
    if vers==0:
        cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    else:
        cluster_df = pd.read_csv(f'cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
    return cluster_df

def load_raw_data(vers):
    if vers ==0:
        return sf.import_json('../../input/initial_subsample_results.json')
    else:
        return sf.import_json('../../input/initial_subsample_triplets_results.json')

def fetch_mainstream_extreme_clusters(hvv, n_clusters, vers):
    cluster_df = load_cluster_df(hvv, n_clusters, vers)
    main_extreme_df = cluster_df.loc[
        (cluster_df['extreme'] >= -0.5) & (cluster_df['mainstream'] >= -0.5) & (cluster_df['time'] <= 0)]
    main_extreme_ids = list(main_extreme_df['cluster_id'].values)
    return main_extreme_ids


def fetch_cluster_components(raw_data , ids, hvv, vers):

    if vers==0:
        clustering = sf.import_pkl_file( f'agglom_clusering_{hvv}_{n_clusters}.pkl')
    else:
        clustering = sf.import_pkl_file(f'agglom_clusering_{hvv}_{n_clusters}_v{vers}.pkl')
    labels = list(clustering.labels_)

    hvv_data = []
    for i in range(len(raw_data)):
        elt = raw_data[i]
        if 'embedding_result' not in elt.keys():
            continue
        if labels[i] not in ids: # TODO: the counting is messed up here
            continue
        for j in range(len(elt['embedding_result'][hvv])):
            if vers==0:
                new_elt = {'_id': elt['_id']["$oid"],
                           'sample_id':elt['sample_id'],
                           'text':elt['denoising_result'][hvv][j],
                           'cluster_id': int(labels[i]),
                           'hvv':hvv}
            else:
                new_elt =  {'_id': elt['_id']["$oid"],
                           'publish_date':elt['publish_date'],
                            'partisanship': elt['partisanship'],
                           'text':elt['denoising_result'][hvv][j],
                           'cluster_id': int(labels[i]),
                           'hvv':hvv}
            hvv_data.append(new_elt)


    #     cluster_id = hvv_data[i]['cluster_id']
    #     if cluster_id in clusters.keys():
    #         clusters[cluster_id].append(hvv_data[i])

    return hvv_data

## Get mainstream extremes for each of the three hvvs ##
# n_clusters = 5000
n_clusters = 2500
VERS =1
data = load_raw_data(VERS)

cluster_components = []
for hvv in ['hero','villain','victim']:
    cluster_ids = fetch_mainstream_extreme_clusters( hvv, n_clusters,VERS)
    cluster_components += fetch_cluster_components(data, cluster_ids,hvv,VERS)

df = pd.DataFrame(data=cluster_components)
df['Immigration'] = False
df['Anti-semitism'] = False
df['Islamophobia'] = False
df['Transphobia'] = False

# h_ids = [item["_id"] for k in cluster_components['hero'].keys() for item in cluster_components['hero'][k]]
# vil_ids = [item["_id"] for k in cluster_components['villain'].keys() for item in cluster_components['villain'][k]]
# vic_ids = [item["_id"] for k in cluster_components['victim'].keys() for item in cluster_components['victim'][k]]

# Query which of these ids have keywords
keyword_loc = "C:\\Users\\khahn\\Documents\\Github\\VolumeAnalysis\\complete_dataset\\output"
keyword_files = sf.get_files_from_folder(keyword_loc,'csv')
unique_art_ids = list(df['_id'].unique())
for file in keyword_files:
    category = file.split('output\\')[1].split('_Keyword')[0]
    data = sf.import_csv(file)
    for elt in data:
        if elt[0] in unique_art_ids:
            rel_index = df.loc[df['_id']==elt[0]].index
            df.loc[rel_index, category] = True

for cat in ['Immigration','Islamophobia','Anti-semitism','Transphobia']:
    for hvv in ['hero','villain','victim']:
        identified = len(df.loc[(df[cat]==True) & (df['hvv']==hvv)])
        print(f"{cat}, {hvv}: {identified}")
print('')