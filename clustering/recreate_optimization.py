import shared_functions as sf
from typing import List
import numpy as np
import pandas as pd
from plotly import express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN

import re
import sklearn

def import_optimization( hvv, cluster_type, visualize=False):
    print('importing optimizations')
    optimizations = sf.get_files_from_folder(f'cluster_experiments/clustering_optimizations','json')
    relevant = [x for x in optimizations if cluster_type in x and hvv in x]
    if len(relevant) ==0:
        print(f"No files found for {hvv}, {cluster_type}")
        return
    data = []
    print('iterating through relevant files')
    for file in relevant:
        content = sf.import_json(file)
        data += content['content'][1:]
        columns = content['content'][0]

    # if hvv=='':
    #     data = sf.import_json(f'cluster_experiments/clustering_optimizations/{cluster_type}_n_optimization_{start}_{end}.json')
    # else:
    #     data = sf.import_json(f'cluster_experiments/clustering_optimizations/{cluster_type}_n_optimization_{start}_{end}_{hvv}.json')
    df = pd.DataFrame(data=data, columns=columns)
    max_ = df[df['silhouette_score'] == df['silhouette_score'].max()]
    print(f"{cluster_type}, {hvv}: Max silhouette score is {df['silhouette_score'].max()}")
    print('doing visualization')
    if cluster_type=='dbscan':
        df = df.sort_values(by='epsilon')
        if visualize:
            x = df['epsilon'].values
            y = df['silhouette_score'].values
            plt.plot(x, y)
            plt.xlabel('Epsilon')
            plt.ylabel('Silhouette Score')
            if cluster_type=='agglom':
                plt.title(f"Agglomerative Clustering, {hvv} archetype")
            else:
                plt.title(f"{cluster_type}, {hvv}")
            plt.show()

        return list(max_['epsilon'].values)[0]
    else:
        df = df.sort_values(by='num_clusters')

        if visualize:
            x = df['num_clusters'].values
            y = df['silhouette_score'].values
            plt.plot(x,y)
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            if cluster_type == 'agglom':
                plt.title(f"Agglomerative Clustering, {hvv} archetype")
            else:
                plt.title(f"{cluster_type}, {hvv}")
            plt.show()

        return list(max_['num_clusters'].values)[0]


def tsne_visualization(texts: List[str], labels: List[str], embeddings: List, title):
    arr = np.array(embeddings)
    X_embedded = TSNE(n_components=2).fit_transform(arr)
    df_embeddings = pd.DataFrame(X_embedded)
    df_embeddings = df_embeddings.rename(columns={0: 'x', 1: 'y'})
    df_embeddings = df_embeddings.assign(label=labels)
    df_embeddings = df_embeddings.assign(text=texts)

    fig = px.scatter(
        df_embeddings,
        x='x',
        y='y',
        color='label',
        labels={'color': 'label'},
        hover_data=['text'],
        title=title
    )
    fig.show()

def fetch_data(filename,hvv='hero')->tuple:
    if filename.endswith('json'):
        data = sf.import_json(filename)
        vectors =[elt for x in data if 'embedding_result'in x.keys() and 'processing_result' in x.keys() for elt in x['embedding_result'][hvv]]
        text = [elt for x in data if 'embedding_result'in x.keys() and 'processing_result' in x.keys() for elt in x['processing_result'][hvv]]
    else:
        data = sf.import_pkl_file(filename)['content']
        vectors = [x['embedding'] for x in data]
        text = [x['sentence'] for x in data]
    return vectors, text

def do_kmeans_clustering(n_clusters, embeddings)->list:
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return list(clustering.labels_)

def do_DBSCAN_clustering(epsilon, embeddings, min_samples=5)->list:
    clustering = DBSCAN(min_samples=min_samples,eps=epsilon).fit(embeddings)
    return list(clustering.labels_)

for cluster_type in ['dbscan','kmeans','agglom']:
    for template in ['combo_a','combo_b']:
        n_clusters = import_optimization(template, cluster_type, visualize=True)
        # if n_clusters is None:
        #     continue
        # if 'a' in template:
        #     vectors, text = fetch_data('cluster_experiments/sbert_embdddings/initial_subsample_a.pkl')
        # else:
        #     vectors, text = fetch_data('cluster_experiments/sbert_embdddings/initial_subsample_b.pkl')
        # if cluster_type=='dbscan':
        #     cluster_labels = do_DBSCAN_clustering(epsilon=n_clusters, embeddings=vectors)
        # else:
        #     cluster_labels = do_kmeans_clustering(n_clusters, vectors)
        # tsne_visualization(texts=text,
        #                    embeddings=vectors,
        #                    labels=cluster_labels,
        #                    title=f'{cluster_type} {template}'
        #                    )


for cluster_type in ['dbscan','kmeans','agglom']:
    for hvv in ['hero','villain','victim']:
        n_clusters = import_optimization(hvv,cluster_type,visualize=True)
        # if n_clusters is None:
        #     continue
        # vectors, text = fetch_data('cluster_experiments/input/initial_subsample_results.json')
        # if cluster_type=='dbscan':
        #     cluster_labels = do_DBSCAN_clustering(epsilon=n_clusters, embeddings=vectors)
        # else:
        #     cluster_labels = do_kmeans_clustering(n_clusters, vectors)
        # tsne_visualization(texts=text,
        #                    embeddings=vectors,
        #                    labels=cluster_labels,
        #                    title=f'{cluster_type} {hvv}'
        #                    )

