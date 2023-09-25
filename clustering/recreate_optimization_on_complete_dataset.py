import shared_functions as sf
from sklearn.cluster import AgglomerativeClustering
import sklearn
import time

def do_agglom_clustering(n_clusters, embeddings)->tuple:
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    # cluster_centers = list(clustering.cluster_centers_)
    silhouette_score = sklearn.metrics.silhouette_score(X=embeddings,labels=clustering.labels_)
    return clustering.labels_, float(silhouette_score)

def fetch_data(hvv):
    complete_files = [x for x in sf.get_files_from_folder('../input/download_data','pkl') if 'complete_sample' in x]
    vectors = []
    for file in complete_files:
        data = sf.import_pkl_file(file)['content']
        vectors += [y for x in data if 'embedding_result' in x.keys() for y in x['embedding_result'][hvv]]
    return vectors

embeddings = fetch_data('hero')
n_clusters = [int(0.2*len(embeddings)),int(0.25*len(embeddings)),int(0.3*len(embeddings))]
output = ['num_clusters','score','duration']
for k in n_clusters:
    a = time.time()
    labels, score = do_agglom_clustering(k, embeddings)
    b = time.time()
    output.append([k,score,b-a])
sf.export_nested_list('complete_dataset_agglom.csv',output)