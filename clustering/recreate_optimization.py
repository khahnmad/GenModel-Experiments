"""
Works for both separate and combined approaches
"""
import shared_functions as sf
from typing import List
import numpy as np
import pandas as pd
from plotly import express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN

def fetch_relevant_files(hvv_temp:str, cluster_type:str,  vers=0, single_combo='single'):
    if vers == 0:
        opt_loc = f'./clustering_optimizations/version_0'
        if single_combo == 'single':
            opt_loc += f'/separate'
        else:
            opt_loc += f'/combined'

        optimizations = sf.get_files_from_folder(opt_loc, 'json')

    else:
        opt_loc = f'./clustering_optimizations/version_{vers}'
        if single_combo == 'single':
            opt_loc += f'/separate'
        else:
            opt_loc += f'/combined'

        optimizations = sf.get_files_from_folder(opt_loc, 'json')

    relevant = [x for x in optimizations if cluster_type in x and hvv_temp in x]
    if len(relevant) == 0:
        print(f"No files found for {hvv_temp}, {cluster_type}, v{vers}")
        return None
    return relevant

def plot_clustering_optimization(df, x_label, hvv_temp, cluster_type):
    x = df[x_label].values
    y = df['silhouette_score'].values
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel('Silhouette Score')
    if cluster_type == 'agglom':
        plt.title(f"Agglomerative Clustering, {hvv_temp} archetype")
    else:
        plt.title(f"{cluster_type}, {hvv_temp}")
    plt.show()



def import_optimization(hvv_temp:str, cluster_type:str, visualize=False, vers=0, single_combo='single'):
    # Fetch the relevant optimization files
    relevant_files = fetch_relevant_files(hvv_temp, cluster_type, vers, single_combo)
    if relevant_files is None:
        return

    # Import the data from the relevant files
    data = []
    for file in relevant_files:
        content = sf.import_json(file)
        data += content['content'][1:]
        columns = content['content'][0]

    # Format the data into a df
    df = pd.DataFrame(data=data, columns=columns)
    max_ = df[df['silhouette_score'] == df['silhouette_score'].max()]
    print(f"{cluster_type}, {hvv_temp}, v{vers}, {single_combo}: Max silhouette score is {df['silhouette_score'].max()}")

    # Plot the data
    if cluster_type=='dbscan':
        x_label = 'epsilon'
    else:
        x_label = 'num_clusters'

    df = df.sort_values(by=x_label)

    if visualize:
        plot_clustering_optimization(df, x_label, hvv_temp, cluster_type)
    return list(max_[x_label].values)[0]


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




###################### ACTION ############################
# COMBINED HVV
for cluster_type in ['dbscan','kmeans','agglom']:
    for template in ['combo_a','combo_b','combo_c','combo_d']:
        for version in range(3):
            n_clusters = import_optimization(hvv_temp=template, cluster_type=cluster_type,vers=version,
                                             visualize=True, single_combo='combo')
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

