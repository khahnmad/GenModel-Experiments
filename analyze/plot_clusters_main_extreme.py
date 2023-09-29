import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_cluster_df(hvv_temp, n_clusters, single_combo, vers=0):
    if vers==0:
        if single_combo=='single':
            cluster_df = pd.read_csv(f'single_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
        else:
            cluster_df = pd.read_csv(f'combo_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}.csv')
    else:
        cluster_df = pd.read_csv(f'single_hvv/cluster_interpretation_{hvv_temp}_{n_clusters}_v{vers}.csv')
    return cluster_df

#
# HVV_TEMP = 'villain'
# N_CLUSTERS = 5000
# SINGLE_COMBO = 'single'

# HVV_TEMP = 'b'
# N_CLUSTERS = 3500
# SINGLE_COMBO = 'combo'

HVV_TEMP = 'villain'
N_CLUSTERS = 2500
SINGLE_COMBO = 'single'
VERS = 1

cluster_df = load_cluster_df(HVV_TEMP, N_CLUSTERS,SINGLE_COMBO,VERS)
cluster_df = cluster_df.dropna()
sorted = cluster_df.sort_values(by='time')
x = list(cluster_df['mainstream'].values)
y = list(cluster_df['extreme'].values)
c = list(cluster_df['time'].values)
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