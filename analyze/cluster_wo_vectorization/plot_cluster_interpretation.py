import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shared_functions as sf


def load_cluster_df(n=5):
    if n == 5:
        data = sf.import_json('alphabet_clusters.json')['content']
        df = pd.DataFrame(data=data[1:], columns=data[0])
    else:
        data = sf.import_json(f'alphabet_clusters_{n}.json')['content']
        df = pd.DataFrame(data=data[1:], columns=data[0])
    return df

cluster_df = load_cluster_df(3)
cluster_df = cluster_df.dropna()
sorted = cluster_df.sort_values(by='slope')
x = list(cluster_df['main'].values)
y = list(cluster_df['extreme'].values)
c = list(cluster_df['slope'].values)
# plt.scatter(x,y,c, cmap='Greens')

color = []
for item in c:
    if item <=0:
        color.append('green')
    else:
        color.append('blue')
fig, ax = plt.subplots()
plt.scatter(x,y,c=color)
circle = plt.Circle((0, 0), 0.5, color='r', fill=False)

plt.plot([-4,2.5],[0,0],color='black')
plt.plot([0,0],[-3.5,2.5],color='black')
plt.xlabel('Mainstream Ratio')
plt.ylabel('Extremist Ratio')
ax.add_patch(circle)
red_patch = mpatches.Patch(color='red', label='Radius of accepted candidates for mainstreamed extremist clusters')
blue_patch = mpatches.Patch(color='blue', label='Positive Slope (moving towards the partisan right)')
green_patch = mpatches.Patch(color='green', label='Negative Slope (moving towards the partisan left)')
plt.legend(handles=[red_patch, blue_patch,green_patch])
plt.title('Clusters by their Mainstream and Extremist Ratios and Partisanship Movement over time')
plt.show()