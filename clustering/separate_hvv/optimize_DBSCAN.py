# import shared_functions as sf
from sklearn.cluster import DBSCAN
import pickle
import re
import sklearn
from typing import List
import time
import json

def export_as_json(export_filename:str, output):
    if export_filename.endswith('.json') is not True:
        raise Exception(f"{export_filename} should be a .json file")
    try:
        with open(export_filename, "w") as outfile:
            outfile.write( json.dumps(output))
    except TypeError:
        with open(export_filename, "w") as outfile:
            outfile.write( json_util.dumps(output))


def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content

def import_pkl_file(file):
    with open(file, "rb") as f:
        pkl_file = pickle.load(f)
        f.close()
    return pkl_file

def fetch_data(filename,hvv='hero')->list:
    if filename.endswith('json'):
        data = import_json(filename)
        vectors = [elt for x in data if 'embedding_result' in x.keys() for elt in x['embedding_result'][hvv]]
    elif filename.endswith('pkl'):
        data = import_pkl_file(filename)
        vectors = [x['embedding'] for x in data]
    else:
        raise KeyError('file should be .json or .pkl format')

    return vectors

def do_DBSCAN_clustering(epsilon, embeddings, min_samples=5)->tuple:
    clustering = DBSCAN(min_samples=min_samples,eps=epsilon).fit(embeddings)
    # cluster_centers = list(clustering.cluster_centers_)
    silhouette_score = sklearn.metrics.silhouette_score(X=embeddings,labels=clustering.labels_)
    return clustering, float(silhouette_score)

def optimize_cluster_number(start_n:int, end_n:int, interval:int, embeddings:list, min_samples=5):
    n_optimization = [['epsilon','min_samples','silhouette_score','duration']]
    for n in range(start_n, end_n+1,interval):
        a = time.time()
        clustering, score = do_DBSCAN_clustering(n, embeddings, min_samples)
        b = time.time()
        n_optimization.append([n, min_samples, score, b-a])
        print(f"{n}/{end_n}")
    return n_optimization

for hvv in ['villain','victim']:
    print('starting')
    embeddings = fetch_data('../../input/initial_subsample_results.json', hvv)
    print('loaded embeddings')
    start_n = 1
    # end_n = int(0.5 * len(embeddings))
    end_n = 20
    interval = 2
    # Experiment w different cluster number

    print('beginning optimization')
    optimization = optimize_cluster_number(start_n, end_n,interval, embeddings)
    export_as_json(f'dbscan_n_optimization_{start_n}_{end_n}_{hvv}.json',{'content':optimization,
                                                                       'metadata':{'start':start_n,
                                                                                   'end':end_n,
                                                                                   'clustering':'agglomerative',
                                                                                   'initial_subsample':True,
                                                                                   'num_embeddings':len(embeddings),
                                                                                   'hvv':hvv,
                                                                                   'min_samples':5}})

    print(f'Exporting {hvv}')