import shared_functions as sf
import numpy as np
import dateutil.parser
from scipy.stats import linregress
import datetime
import pandas as pd
import time

def fetch_parts_dates(ids, df):
    return list(df.loc[ids]['partisanship'].values), list(df.loc[ids]['publish_date'].values)

def calc_mainstream_odds(parts):
    center = len([x for x in parts if 'Center' in x])
    not_center = len(parts)-center

    p_center = center/len(parts)
    p_n_center = not_center / len(parts)
    if p_n_center==0:
        odds = np.inf
    else:
        odds = np.log(p_center/p_n_center)
    return odds


def calc_extremist_odds(parts):
    extreme = len([x for x in parts if 'FarRight' in x])
    not_extreme = len(parts) - extreme

    p_extreme = extreme / len(parts)
    p_n_extreme = not_extreme / len(parts)

    if p_n_extreme==0:
        odds = np.inf
    else:
        odds = np.log(p_extreme / p_n_extreme)

    return odds

def calc_time_vector(parts, dates):
    conversion = {'FarLeft':-3,"Left":-2,"CenterLeft":-1,"Center":0,
                  "FarRight": 3, "Right":2,"CenterRight":1}
    none_dates = [i for i in range(len(dates)) if dates[i]==None]
    dates = [datetime.datetime.timestamp(dateutil.parser.parse(x)) for x in dates if x!=None]
    parts = [conversion[parts[i]] for i in range(len(parts)) if i not in none_dates]
    try:
        line = linregress(dates, parts)
        slope = line.slope
    except ValueError:
        slope = None
    return slope

def remove_duplicate_ids(ids, df):
    wout_duplicates = []
    for id_set in ids:
        art_ids = df.iloc[id_set]['id'].unique()
        # art_ids = list(df.iloc[ids]['id'].values)
        # art_ids = [x["$oid"] for x in art_ids]
        # updated = df.loc[df['id'] in art_ids].index
        if len(art_ids)>=n:
            # updated = df[df['id'].isin(art_ids)].index
            wout_duplicates.append(id_set)
    return wout_duplicates


def fetch_text(ids, df):
    return df.loc[ids[0]]['hero'], df.loc[ids[0]]['villain'], df.loc[ids[0]]['victim']

def get_clusters(file):
    data = sf.import_json(file)['content']

    df = pd.DataFrame(data,
                      columns=['hero','villain','victim','id','partisanship','publish_date'])
    df['id'] = df['id'].apply(lambda x: x["$oid"])
    df = df.drop_duplicates()
    grouped = df.groupby(by=['hero','villain','victim'])
    groups = grouped.groups
    cluster_indices = [groups[k] for k in groups.keys() if len(groups[k])>=n]
    # cluster_indices = remove_duplicate_ids(cluster_indices, df)

    interpretation = []
    for ids in cluster_indices:

        hero,villain,victim = fetch_text(ids, df)
        partisanships, publish_dates = fetch_parts_dates(ids, df)

        main = calc_mainstream_odds(partisanships)
        extreme = calc_extremist_odds(partisanships)
        slope = calc_time_vector(parts=partisanships,dates=publish_dates)

        interpretation.append([hero,villain,victim,main,extreme,slope])
    return interpretation


n = 3

files = sf.get_files_from_folder('alphabet','json')
# prev_data = sf.import_json('alphabet_clusters.json')
# interpretation = prev_data['content']
# last_file = prev_data['metadata']['last_file']
# last_file_id = [i for i in range(len(files)) if files[i]==last_file][0]
# start_id = last_file_id+1
interpretation = [['hero','villain','victim','main','extreme','slope']]
start_id = 0
for i in range(start_id,len(files)):
    file = files[i]
    print(f'running {file}')
    a = time.time()
    interpretation += get_clusters(file)
    b = time.time()
    sf.export_as_json('alphabet_clusters_3.json', {'content':interpretation,
                                                 'metadata': {'last_file':file}})
    print(f'   Took {(b-a)/60} minutes, interpretaion is {len(interpretation)} rows')
