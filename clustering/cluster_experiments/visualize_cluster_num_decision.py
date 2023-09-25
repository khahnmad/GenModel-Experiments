import matplotlib.pyplot as plt
import shared_functions as sf
import pandas as pd

# Initial subsample only so far, but should do with full as well
agglom_files = sf.get_files_from_folder('clustering_optimizations','json')
hvvs = ['hero','villain','victim']
num_embeddings = {'villain':22743,'hero':22743,'victim':22743}
for hvv in hvvs:
    relevant = [x for x in agglom_files if hvv in x and 'agglom' in x]
    if len(relevant) ==0:
        print(f"No files found for {hvv}")
        continue

    data = []
    for file in relevant:
        content = sf.import_json(file)
        data += content['content'][1:]
        columns = content['content'][0]

    df = pd.DataFrame(data=data, columns=columns)
    df = df.sort_values(by='num_clusters')
    df['percent_data'] = df['num_clusters']/num_embeddings[hvv]

    x = list(df['percent_data'].values)
    y = list(df['silhouette_score'].values)

    plt.plot(x,y,label=hvv)
plt.legend()
plt.xlabel('% of data')
plt.ylabel('Silhouette Score')
plt.show()

# x axis : % of data
# y axis: silhouette score
# by hvv