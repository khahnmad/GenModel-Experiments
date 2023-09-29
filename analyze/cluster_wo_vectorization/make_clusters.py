import shared_functions as sf
import numpy as np
import dateutil.parser
from scipy.stats import linregress
import datetime
import pandas as pd
import time


def calc_mainstream_odds(data):
    parts = [x[1] for x in data]
    center = len([x for x in parts if 'Center' in x])
    not_center = len(parts)-center

    p_center = center/len(parts)
    p_n_center = not_center / len(parts)
    if p_n_center==0:
        odds = np.inf
    else:
        odds = np.log(p_center/p_n_center)
    return odds


def calc_extremist_odds(data):
    parts = [x[1] for x in data]
    extreme = len([x for x in parts if 'FarRight' in x])
    not_extreme = len(parts) - extreme

    p_extreme = extreme / len(parts)
    p_n_extreme = not_extreme / len(parts)

    if p_n_extreme==0:
        odds = np.inf
    else:
        odds = np.log(p_extreme / p_n_extreme)

    return odds

def time_vector(data):
    conversion = {'FarLeft':-3,"Left":-2,"CenterLeft":-1,"Center":0,
                  "FarRight": 3, "Right":2,"CenterRight":1}
    dates = [datetime.datetime.timestamp(dateutil.parser.parse(x[-1])) for x in data if x[-1]!=None]
    parts = [conversion[x[1]] for x in data if x[-1]!=None]
    try:
        line = linregress(dates, parts)
        slope = line.slope
    except ValueError:
        slope = None
    return slope


def make_clusters(title, data, cluster_size):
    outcome = [['cluster_id','hvv','mainstream','extremist','slope']]
    cluster_count = 0
    h_count, vil_count = 0,0
    start = time.time()
    # for h in data.keys():
    #     print(f"H: {h_count} / {len(data)}")
    #     h_count+=1
    #
    #     for vil in data[h].keys():
    #         # print(f"   Vil: {vil_count} / {len(data[h])}")
    #         vil_count+=1
    #         for vic in data[h][vil].keys():
    #             if len(data[h][vil][vic])<cluster_size:
    #                 continue
    #             main = calc_mainstream_odds(data[h][vil][vic])
    #             extreme = calc_extremist_odds(data[h][vil][vic])
    #             time_v = time_vector(data[h][vil][vic])
    #             outcome.append([cluster_count, f"{h}</>{vil}</>{vic}",main, extreme, time_v])
    #             cluster_count+=1
    #     end = time.time()
    #     print(f"   {end-start} seconds")
    #     start = end
    sf.export_nested_list(f'{title}_clusters_{cluster_size}.csv', outcome)


def find_mainstreamed_extremist_narratives(title,num):
    df = pd.read_csv(f'{title}_clusters_{num}.csv')
    df['mainstream'] = df['mainstream'].astype(float)
    df['extremist'] = df['extremist'].astype(float)
    mainstreamed_extremist_narratives = df.loc[(df['extremist']>=-0.1) & (df['mainstream']>=-0.1) & (df['slope']<=0)]
    print(mainstreamed_extremist_narratives)

# # Exact Matches
# exact_clusters = sf.import_json('exact_matches_7500.json')['content']
# make_clusters('exact_matches', exact_clusters)
# find_mainstreamed_extremist_narratives('exact_matches')

# Cleaned Matches
cleaned_clusters = sf.import_json('cleaned_exact_matches_4500.json')['content']
df = pd.DataFrame(cleaned_clusters)
make_clusters('cleaned_matches', cleaned_clusters,5)
find_mainstreamed_extremist_narratives('cleaned_matches',7)
