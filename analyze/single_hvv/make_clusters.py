import json
import sklearn
from sklearn.cluster import AgglomerativeClustering
import shared_functions as sf
import pandas as pd
import numpy as np
import dateutil.parser
from scipy.stats import linregress
import datetime
import  matplotlib.pyplot as plt

def import_data(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content

def do_clustering(hvv, data, n_clusters, vers=0):
    indices = []
    embeddings = []
    text = []

    for i in range(len(data)):
        elt = data[i]
        if 'embedding_result' not in data[i].keys():
            continue
        for j in range(len(data[i]['embedding_result'][hvv])):
            indices.append(elt['_id']["$oid"])
            embeddings.append(elt['embedding_result'][hvv][j])
            text.append(elt['denoising_result'][hvv][j])

    filename = f'agglom_clusering_{hvv}_{n_clusters}_v{vers}.pkl'
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
        if vers==1:
            clusters[labels[i]].append([indices[i], data_point['publish_date'], data_point['partisanship'], text[i]])
        else:
            clusters[labels[i]].append([indices[i],data_point['sample_id'], text[i]])
    return clusters

def calculate_mainstream(cluster:list,vers=0):
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
    if vers == 0:
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
    time_index = {'2016_1': 0, '2016_2': 1, '2016_3': 2, '2016_4': 3, '2016_5': 4, '2016_6': 5, '2016_7': 6, '2016_8': 7, '2016_9': 8, '2016_10': 9, '2016_11': 10, '2016_12': 11, '2017_1': 12, '2017_2': 13, '2017_3': 14, '2017_4': 15, '2017_5': 16, '2017_6': 17, '2017_7': 18, '2017_8': 19, '2017_9': 20, '2017_10': 21, '2017_11': 22, '2017_12': 23, '2018_1': 24, '2018_2': 25, '2018_3': 26, '2018_4': 27, '2018_5': 28, '2018_6': 29, '2018_7': 30, '2018_8': 31, '2018_9': 32, '2018_10': 33, '2018_11': 34, '2018_12': 35, '2019_1': 36, '2019_2': 37, '2019_3': 38, '2019_4': 39, '2019_5': 40, '2019_6': 41, '2019_7': 42, '2019_8': 43, '2019_9': 44, '2019_10': 45, '2019_11': 46, '2019_12': 47, '2020_1': 48, '2020_2': 49, '2020_3': 50, '2020_4': 51, '2020_5': 52, '2020_6': 53, '2020_7': 54, '2020_8': 55, '2020_9': 56, '2020_10': 57, '2020_11': 58, '2020_12': 59, '2021_1': 60, '2021_2': 61, '2021_3': 62, '2021_4': 63, '2021_5': 64, '2021_6': 65, '2021_7': 66, '2021_8': 67, '2021_9': 68, '2021_10': 69, '2021_11': 70, '2021_12': 71, '2022_1': 72, '2022_2': 73, '2022_3': 74, '2022_4': 75, '2022_5': 76, '2022_6': 77, '2022_7': 78, '2022_8': 79, '2022_9': 80, '2022_10': 81, '2022_11': 82, '2022_12': 83}
    # dates = [create_timestamp(year=int("20"+elt[1].split('.')[0]),
    #                            month=int(elt[1].split('.')[1]))
    #          for elt in cluster]

    if vers==0:
        dates = [time_index[f"{'20'+elt[1].split('.')[0]}_{elt[1].split('.')[1]}"] for elt in cluster]
        parts = [conversion[x[1].split('.')[-1]] for x in cluster]
    else:
        dates = [datetime.datetime.timestamp(dateutil.parser.parse(x[1])) for x in cluster if x[1] != None]
        parts = [conversion[x[2]] for x in cluster]
    try:
        line = linregress(dates, parts)
        slope = line.slope

        # plt.scatter(dates, parts)
        # plt.plot(dates, line.intercept + line.slope*np.array(dates), 'r', label='fitted line')
        # plt.show()
    except ValueError:
        slope = None

    return slope

def calculate_cluster_significance(clusters,vers=0):
    output =[ ['cluster_id','mainstream','extreme','time']]
    for c in clusters.keys():
        main = calculate_mainstream(clusters[c],vers)
        extreme = calculate_extreme(clusters[c],vers)
        time_vector = calculate_time_vector(clusters[c],vers)
        output.append([c,main,extreme, time_vector])
    df = pd.DataFrame(data=output[1:],columns=output[0])
    return df

N_CLUSTERS = 2500
HVV = 'hero'
VERS=1
# Import data
# data = import_data('../../input/initial_subsample_results.json')
data = import_data('../../input/initial_subsample_triplets_results.json')
# try:
#     cluster_df = pd.read_csv(f'cluster_interpretation_{HVV}_{N_CLUSTERS}.csv')
# except FileNotFoundError:
    # Do clustering # also export the clustering
clustering = do_clustering(HVV,data,n_clusters=N_CLUSTERS, vers=VERS)

# Iterate through clusters and calculate mainstream and extreme log odds
cluster_df = calculate_cluster_significance(clustering,vers=1)
cluster_df.to_csv(f'cluster_interpretation_{HVV}_{N_CLUSTERS}_v{VERS}.csv')

# Identify mainstreamed extremist hvvs
main_extreme = cluster_df.loc[(cluster_df['extreme']>=-0.5) & (cluster_df['mainstream']>=-0.5) & (cluster_df['time']<0)]
print('check')

