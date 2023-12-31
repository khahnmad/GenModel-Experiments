import pickle
import sklearn
from sklearn.cluster import AgglomerativeClustering
import shared_functions as sf
import pandas as pd
import numpy as np
import dateutil.parser
from scipy.stats import linregress
import datetime

def import_data(file):
    with open(file, "rb") as f:
        pkl_file = pickle.load(f)
        f.close()
    return pkl_file['content']

def do_clustering(template, data, n_clusters, vers):
    indices = [x['_id']["$oid"] for x in data if 'embedding' in x.keys()]
    embeddings = [x['embedding'] for x in data  if 'embedding' in x.keys()]
    text = [x['sentence'] for x in data if 'embedding' in x.keys()]

    if vers==0:
        filename = f'agglom_clusering_{template}_{n_clusters}.pkl'
    else:
        filename = f'agglom_clusering_{template}_{n_clusters}_v{vers}.pkl'
    try:
        clustering = sf.import_pkl_file(filename)
    except FileNotFoundError:
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
        sf.export_as_pkl(filename, clustering)

    labels = list(clustering.labels_)
    clusters = {c:[] for c in range(n_clusters)}
    for i in range(len(indices)):
        data_point = [x for x in data if x['_id']["$oid"]==indices[i]][0]
        # also add text
        if vers==0:
            clusters[labels[i]].append([indices[i],data_point['sample_id'], text[i]])
        else:
            clusters[labels[i]].append([indices[i], data_point['publish_date'], data_point['partisanship'], text[i]])
    return clusters


def calculate_mainstream(cluster:list, vers=0):
    if vers==0:
        parts = [x[1].split('.')[-1] for x in cluster]
    else:
        parts = [x[2] for x in cluster]
    center = len([x for x in parts if 'Center' in x])
    not_center = len(parts) - center

    p_center = center / len(parts)
    p_n_center = not_center / len(parts)
    if p_n_center == 0:
        odds = np.inf
    else:
        odds = np.log(p_center / p_n_center)
    return odds


def calculate_extreme(cluster:list, vers=0):
    if vers==0:
        parts = [x[1].split('.')[-1] for x in cluster]
    else:
        parts = [x[2] for x in cluster]
    extreme = len([x for x in parts if 'FarRight' in x])
    not_extreme = len(parts) - extreme

    p_extreme = extreme / len(parts)
    p_n_extreme = not_extreme / len(parts)

    if p_n_extreme==0:
        odds = np.inf
    else:
        odds = np.log(p_extreme / p_n_extreme)

    return odds


def create_timestamp(month, year):
    dt = datetime.datetime(month=month, year=year,day=1)
    return datetime.datetime.timestamp(dt)


def calculate_time_vector(cluster, vers=0):
    conversion = {'FarLeft':-3,"Left":-2,"CenterLeft":-1,"Center":0,
                  "FarRight": 3, "Right":2,"CenterRight":1}
    if vers==0:
        dates = [create_timestamp(year=int("20"+elt[1].split('.')[0]),
                                   month=int(elt[1].split('.')[1]))
                 for elt in cluster]
        parts = [conversion[x[1].split('.')[-1]] for x in cluster]
    else:
        dates = [datetime.datetime.timestamp(dateutil.parser.parse(x[1])) for x in cluster if x[1] != None]
        parts = [conversion[x[2]] for x in cluster]
    try:
        line = linregress(dates, parts)
        slope = line.slope
    except ValueError:
        slope = None
    return slope

def calculate_cluster_significance(clusters,vers):
    output =[ ['cluster_id','mainstream','extreme','time']]
    for c in clusters.keys():
        main = calculate_mainstream(clusters[c],vers)
        extreme = calculate_extreme(clusters[c],vers)
        time_vector = calculate_time_vector(clusters[c],vers)
        output.append([c,main,extreme, time_vector])
    df = pd.DataFrame(data=output[1:],columns=output[0])
    return df

# N_CLUSTERS = 3500
# TEMPLATE = 'b'
N_CLUSTERS = 2500
TEMPLATE = 'c'
VERS = 1
# Import data
# data = import_data(f'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\clustering\\cluster_experiments\\'
#                    f'sbert_embdddings\\initial_subsample_{TEMPLATE}.pkl')
data = import_data(f'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\clustering\\initial_subsample_c_v1.pkl')
try:
    cluster_df = pd.read_csv(f'cluster_interpretation_{TEMPLATE}_{N_CLUSTERS}_v{VERS}.csv')
except FileNotFoundError:
    # Do clustering # also export the clustering
    clustering = do_clustering(TEMPLATE,data,n_clusters=N_CLUSTERS, vers=VERS)

    # Iterate through clusters and calculate mainstream and extreme log odds
    cluster_df = calculate_cluster_significance(clustering,VERS)
    cluster_df.to_csv(f'cluster_interpretation_{TEMPLATE}_{N_CLUSTERS}_v{VERS}.csv')

# Identify mainstreamed extremist hvvs
main_extreme = cluster_df.loc[(cluster_df['extreme']>=-0.25) & (cluster_df['mainstream']>=-0.25) & (cluster_df['time']<=0)]
print('check')