from sklearn.cluster import AgglomerativeClustering
import pickle
import re
import sklearn
from typing import List
import time
import json
import glob


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
        data = import_pkl_file(filename)['content']
        vectors = [x['embedding'] for x in data]
    else:
        raise KeyError('file should be .json or .pkl format')

    return vectors

def do_agglom_clustering(n_clusters, embeddings)->tuple:
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    # cluster_centers = list(clustering.cluster_centers_)
    silhouette_score = sklearn.metrics.silhouette_score(X=embeddings,labels=clustering.labels_)
    return clustering, float(silhouette_score)

def optimize_cluster_number(start_n:int, end_n:int, interval:int, embeddings:list):
    n_optimization = [['num_clusters','silhouette_score','duration']]
    for n in range(start_n, end_n+1,interval):
        a = time.time()
        clustering, score = do_agglom_clustering(n, embeddings)
        b = time.time()
        n_optimization.append([n, score, b-a])
        print(f"{n}/{end_n}")
    return n_optimization

def get_files_from_folder(folder_name:str, file_endings:str)->list:
    return [x for x in glob.glob(folder_name + f"/*.{file_endings}")]

def print_missing_n(template):
    files = [x for x in get_files_from_folder('cluster_experiments/sbert_clustering_optimization','json')
             if 'agglom' in x and f"combo_{template}" in x]
    complete = {k:False for k in range(100,6000,100)}
    for file in files:
        numbers= file.split('optimization_')[1].split(f'_combo')[0]
        start = int(numbers.split('_')[0])
        end = int(numbers.split('_')[1])
        for n in range(start, end+100, 100):
            complete[n]=True
    missing = [x for x in complete.keys() if complete[x]==False]
    print(missing)

################## SBERT EXPERIMENT #########################################
template = 'c'
print_missing_n(template)
hvv='combo'
print('starting')
embeddings = fetch_data(f'initial_subsample_{template}_v1.pkl', hvv)
print('loaded embeddings')
start_n = 2000
# end_n = int(0.5 * len(embeddings))
end_n = 2500
interval = 100

optimization = optimize_cluster_number(start_n, end_n, interval,embeddings)
export_as_json(f'cluster_experiments/sbert_clustering_optimization/agglom_n_optimization_{start_n}_{end_n}_combo_{template}_v1.json',{'content':optimization,
                                                                    'metadata':{'start':start_n,
                                                                            'end':end_n,
                                                                            'clustering':'agglomerative',
                                                                            'initial_subsample':True,
                                                                            'num_embeddings':len(embeddings),
                                                                            'hvv':'combo',
                                                                            'template':template,
                                                                                'version':1}})


