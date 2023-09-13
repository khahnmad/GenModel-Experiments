import shared_functions as sf
import matplotlib.pyplot as plt
import pandas as pd

# cleaned_data = sf.import_csv('cleaned_matches_clusters_7.csv')
# exact = sf.import_csv('exact_matches_clusters.csv')

# clean_df = pd.read_csv('cleaned_matches_clusters_7.csv')
# clean_df['mainstream'] = clean_df['mainstream'].astype(float)
# clean_df['extremist'] = clean_df['extremist'].astype(float)

exact_df = pd.read_csv('exact_matches_clusters.csv')
exact_df['mainstream'] = exact_df['mainstream'].astype(float)
exact_df['extremist'] = exact_df['extremist'].astype(float)


x = list(exact_df['extremist'].values)
y = list(exact_df['mainstream'].values)
slope = list(exact_df['slope'].values)
color = []
for item in slope:
    if item <=0:
        color.append('red')
    else:
        color.append('blue')
plt.scatter(x,y,c=color)
plt.xlabel('Extremist')
plt.ylabel('Mainstream')
plt.title('Red: neg slope')
plt.show()
