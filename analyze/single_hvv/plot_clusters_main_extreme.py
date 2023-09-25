import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
HVV = 'hero'
N_CLUSTERS = 5000

cluster_df = pd.read_csv(f'cluster_interpretation_{HVV}_{N_CLUSTERS}.csv')
cluster_df = cluster_df.dropna()
sorted = cluster_df.sort_values(by='time')
x = list(cluster_df['mainstream'].values)
y = list(cluster_df['extreme'].values)
c = list(cluster_df['time'].values)
# plt.scatter(x,y,c, cmap='Greens')

color = []
for item in c:
    if item <=0:
        color.append('red')
    else:
        color.append('blue')
plt.scatter(x,y,c=color)


plt.plot([-4,2.5],[0,0],color='black')
plt.plot([0,0],[-3.5,2.5],color='black')
plt.xlabel('Mainstream Ratio')
plt.ylabel('Extremist Ratio')
red_patch = mpatches.Patch(color='red', label='Negative Slope (moving towards the partisan left)')
blue_patch = mpatches.Patch(color='blue', label='Positive Slope (moving towards the partisan right)')
plt.legend(handles=[red_patch, blue_patch])
plt.title('Hero Clusters by their Mainstream and Extremist Ratios and Partisanship Movement over time')
plt.show()