from typing import List
import shared_functions as sf
import pandas as pd
from string import punctuation
import datetime

PARTISANSHIPS = ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right', 'FarRight']
NUM_BINS = 171

def remove_punct(text):
    for char in punctuation:
        text = text.replace(char,'')
    return text

def fetch_data():
    df = pd.read_csv('subsample_cleaned_hvv_data.csv')
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    df['clean_content'] = df.apply(lambda x: remove_punct(x['content'].lower()) if isinstance(x['content'],str) else None, axis=1)

    df['time_index'] = None
    start = datetime.datetime(year=2016, month=1, day=1)
    for i in range(1,NUM_BINS+1):

        end = start + datetime.timedelta(days=15)
        df.loc[(df['time'] >= start) & (df['time'] < end), 'time_index'] = i
        start = end
    return df

def get_hvv_input_combos(length:str)->List[str]:
    hvv = ['hero', 'villain', 'victim']
    if length=='single':
        return hvv
    elif length=='double':
        output = []
        for i in range(len(hvv)):
            for j in range(len(hvv)):
                if i!=j and j>i:
                    output.append(f"{hvv[i]} {hvv[j]}")
        return output
    else:
        return ['hero villain victim']


def get_unique_fr_appearance_date(hvv_input:str, df)->List:
    if hvv_input.count(' ')==0:
        clean_content = list(df.loc[(df['partisanship']=='FarRight') & (df['hvv']==hvv_input)]['clean_content'].unique())
        app_dates = []
        for item in clean_content:
            v = df.loc[(df['partisanship']=='FarRight') & (df['hvv']==hvv_input) & (df['clean_content']==item)]['time_index'].min()
            prev_occurrence = df.loc[(df['clean_content']==item) & (df['hvv']==hvv_input) & (df['time_index']<v)]
            if len(prev_occurrence) <1:
                app_dates.append([item,v])
    elif hvv_input.count(' ')==1:
        a = hvv_input.split(' ')[0]
        b = hvv_input.split(' ')[1]
        a_content = list(df.loc[(df['partisanship']=='FarRight') & (df['hvv']==a)]['clean_content'].unique())
        combos = []
        for item in a_content:
            art_ids = list(df[(df['clean_content']==item) & (df['hvv']==a)]['article_id'].unique())
            for a_id in art_ids:
                b_values = list(df[(df['article_id']==a_id) & (df['hvv']==b)]['clean_content'].unique())
                for b_val in b_values:
                    combos.append(f"{item}</>{b_val}")

            # app_dates.append([item, v])
    return app_dates



def find_mainstreamed_appearances(hvv_input, candidates:List, threshold)->list:
    # Determine output based on what successes actually look like
    success = []
    for cand in candidates:
        apps = df.loc[(df['clean_content']==cand[0]) & (df['time_index']>cand[1])  & (df['hvv']==hvv_input)]
        if len(apps)==0:
            continue
        perc_centrist = len([x for x in list(apps['partisanship'].values) if 'Center' in x])/len(list(apps['partisanship'].values))
        if perc_centrist > threshold:
            success.append([cand[0], apps[['partisanship','time','time_index']]])

    return success


# if __name__ == '__main__':
#     mainstreaming_threshold = 0.5
#     df = fetch_data()
#     for combo_length in ['single','double','triple']:
#         hvv_combos = get_hvv_input_combos(combo_length)
#
#         for hvv_combo in hvv_combos:
#             appearance_dates = get_unique_fr_appearance_date(hvv_combo,df)
#             successful_paths = find_mainstreamed_appearances(hvv_combo, appearance_dates,mainstreaming_threshold)

def get_hvv_combos(data, output):
    heros = sf.remove_duplicates([remove_punct(x.lower()) for x in data['processing_result']['hero']])
    villains = sf.remove_duplicates([remove_punct(x.lower()) for x in data['processing_result']['villain']])
    victims = sf.remove_duplicates([remove_punct(x.lower()) for x in data['processing_result']['victim']])

    for i in heros:
        for j in villains:
            for k in victims:
                # if f"{i}</>{j}</>{k}" not in output.keys():
                #     output[f"{i}</>{j}</>{k}"] = []
                output[f"{i}</>{j}</>{k}"].append([data['partisanship'],data['publish_date'],data["_id"]["$oid"]])
    return output

data = sf.import_json('../output/sample_processing_results.json')['content']
hvv_appearances = {}
for i in range(len(data)):
    elt = data[i]
    hvv_appearances = get_hvv_combos(elt,hvv_appearances)
    if str(i).endswith('00'):
        sf.export_as_json('sample_hvv_appearances.json',hvv_appearances)
sf.export_as_json('sample_hvv_appearances.json',hvv_appearances)
