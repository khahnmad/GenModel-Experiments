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

def fetch_data(filename)->list:
    data = import_pkl_file(filename)['content']
    vectors = [x['embedding'] for x in data]
    return vectors

def do_DBSCAN_clustering(epsilon, embeddings, min_samples=5)->tuple:
    clustering = DBSCAN(min_samples=min_samples,eps=epsilon).fit(embeddings)
    # cluster_centers = list(clustering.cluster_centers_)
    try:
        silhouette_score = float(sklearn.metrics.silhouette_score(X=embeddings,labels=clustering.labels_))
    except ValueError:
        silhouette_score = None
    return clustering, silhouette_score

def optimize_cluster_number(start_n:int, end_n:int, interval:int, embeddings:list, min_samples=5):
    n_optimization = [['epsilon','min_samples','silhouette_score','duration']]
    for n in range(start_n, end_n+1,interval):
        a = time.time()
        clustering, score = do_DBSCAN_clustering(n, embeddings, min_samples)
        b = time.time()
        n_optimization.append([n, min_samples, score, b-a])
        print(f"{n}/{end_n}")
    return n_optimization

def print_missing_n(template):
    files = [x for x in sf.get_files_from_folder('cluster_experiments/clustering_optimizations','json')
             if 'dbscan' in x and f"combo_{template}" in x]
    complete = {k:False for k in range(100,12000,100)}
    for file in files:
        numbers= file.split('optimization_')[1].split(f'_{hvv}')[0]
        start = int(numbers.split('_')[0])
        end = int(numbers.split('_')[1])
        for n in range(start, end+100, 100):
            complete[n]=True
    missing = [x for x in complete.keys() if complete[x]==False]
    print(missing)


hvv='combo'
for version in range(1,3):
    for template in ['a','b','c']:
# template = 'c'
# version =0

        print(f'starting {template}, v{version}')
        # Fetch embeddings
        if version==0:
            embeddings = fetch_data(f'sbert_embeddings/initial_subsample_{template}.pkl')
        else:
            embeddings = fetch_data(f'sbert_embeddings/initial_subsample_{template}_v{version}.pkl')

        # Set epsilon parameters
        start_n = 1
        end_n = 20
        interval = 1

        # Run optimization
        optimization = optimize_cluster_number(start_n, end_n,interval, embeddings)
        export_as_json(f'../clustering_optimizations/version_{version}/combined/dbscan_n_optimization_{start_n}_{end_n}_combo_{template}.json',
                       {'content':optimization,
                        'metadata':{'start':start_n,
                                    'end':end_n,
                                    'clustering':'dbscan',
                                    'initial_subsample':True,
                                    'num_embeddings':len(embeddings),
                                    'hvv':hvv,
                                    'min_samples':5,
                                    'template':template}})

        print(f'Exported')