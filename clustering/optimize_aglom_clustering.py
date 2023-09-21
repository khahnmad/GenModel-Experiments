import shared_functions as sf
from sklearn.cluster import AgglomerativeClustering
import pickle
import re
import sklearn
from typing import List
import time

def fetch_data(filename,hvv='hero')->list:
    if filename.endswith('json'):
        data = sf.import_json(filename)
        vectors = [elt for x in data if 'embedding_result' in x.keys() for elt in x['embedding_result'][hvv]]
    elif filename.endswith('pkl'):
        data = sf.import_pkl_file(filename)
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

################## SBERT EXPERIMENT #########################################
#
# embeddings = fetch_data('sentence_embeddings_test.pkl')
#
# start_n = 2
# end_n = int(0.5 * len(embeddings))
# # Experiment w different cluster number
#
#
# optimization = optimize_cluster_number(start_n, end_n,embeddings)
# sf.export_as_json(f'kmeans_n_optimization_{start_n}_{end_n}.json',{'content':optimization,
#                                                                    'metadata':{'start':start_n,
#                                                                                'end':end_n,
#                                                                                'clustering':'kmeans',
#                                                                                'initial_subsample':True,
#                                                                                'num_embeddings':len(embeddings),
#                                                                                'hvv':'combo'}})


############### HERO EMBEDDINGS EXPERIMENT #######################################
for hvv in ['villain','victim','hero']:
    print('starting')
    embeddings = fetch_data('cluster_experiments/input/initial_subsample_results.json', hvv)
    print('loaded embeddings')
    start_n = 7500
    # end_n = int(0.5 * len(embeddings))
    end_n = 8000
    interval = 100
    # Experiment w different cluster number

    print('beginning optimizaation')
    optimization = optimize_cluster_number(start_n, end_n,interval, embeddings)
    sf.export_as_json(f'agglom_n_optimization_{start_n}_{end_n}_{hvv}.json',{'content':optimization,
                                                                       'metadata':{'start':start_n,
                                                                                   'end':end_n,
                                                                                   'clustering':'agglomerative',
                                                                                   'initial_subsample':True,
                                                                                   'num_embeddings':len(embeddings),
                                                                                   'hvv':hvv}})

    print(f'Exporting {hvv}')