"""
Questions:
- how many hvvs identified by partisanship, time, category
- most common hvvs by partisanship, time, category

- hvvs unique to Far RIght, Far Left
- hvvs shared by Far Right, Far left
- hvvs unique to the right wing
- hvvs unique to the left wing

- match with context window data
"""
import shared_functions as sf
import pandas as pd
from collections import Counter
import datetime
import matplotlib.pyplot as plt

partisanships = ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right', 'FarRight']
num_bins = 171

def fetch_data(test_data:bool=False)->list:
    if test_data:
        data = sf.import_json('../output/testing_data/complete/google_flan-t5-base_B_langchain_annotations.json')['content']
    else:
        data = sf.import_json('../output/subsample_processing_results.json')['content']
    return data

# def remove_sep_tokens(data:str):
#     return data.replace('<pad>','').replace('</s>','')

def clean_data(test_data=False):
    data = fetch_data(test_data)

    content = [['partisanship','time','article_id','content','hvv']]
    for elt in data:
        for hvv in ['hero','villain','victim']:
            if 'processing_result' not in elt.keys():
                continue
            if isinstance(elt['processing_result'][hvv],str):
                # clean_elt = remove_sep_tokens(elt['processing_result'][hvv])
                content.append([elt['partisanship'],elt['publication_date'],elt['article_id'],elt['processing_result'][hvv],hvv])
            else:
                # already = []
                for subelt in elt['processing_result'][hvv]:
                    # clean_elt = remove_sep_tokens(subelt)
                    # if clean_elt in already:
                    #     continue
                    # already.append(clean_elt)
                    content.append([elt['partisanship'], elt['publish_date'], elt['_id']["$oid"], subelt, hvv])
    sf.export_nested_list('subsample_cleaned_hvv_data.csv',content)

def find_most_common_per_partisanship(hvv, partisanship, df,n):
    if hvv!='all':
        relevant = list(df.loc[(df['partisanship']==partisanship)&(df['hvv']==hvv)]['content'].values)
    else:
        relevant = list(df.loc[(df['partisanship']==partisanship)]['content'].values)
    counter = Counter(relevant)
    return counter.most_common(n)

def find_most_common_per_time(hvv, month, year, df,n):
    start_time = datetime.datetime(month=month,year=year,day=1)
    if month==12:
        end_time = datetime.datetime(month=1, year=year+1, day=1) - datetime.timedelta(seconds=1)
    else:
        end_time =  datetime.datetime(month=month+1,year=year,day=1)- datetime.timedelta(seconds=1)

    if hvv!='all':
        relevant = list(df.loc[(df['time']>=start_time)& (df['time']<=end_time)&(df['hvv']==hvv)]['content'].values)
    else:
        relevant = list(df.loc[(df['time']>=start_time)& (df['time']<=end_time)]['content'].values)
    counter = Counter(relevant)
    return counter.most_common(n)

def find_hvv_combinations(h,vil,vic):
    combos = []
    for i in h:
        for j in vil:
            for k in vic:
                combos.append(f"{i}</>{j}</>{k}")
    return combos

def find_most_common_triples(partisanship,n):
    unique_ids = list(df['article_id'].unique())
    hvv_combos = []
    for u_id in unique_ids:
        heroes = list(df.loc[(df['article_id']==u_id) & (df['partisanship'] == partisanship) & (df['hvv'] == 'hero')]['content'])
        villains = list(df.loc[(df['article_id']==u_id) & (df['partisanship'] == partisanship) & (df['hvv'] == 'villain')]['content'])
        victims = list(df.loc[(df['article_id']==u_id) & (df['partisanship'] == partisanship) & (df['hvv'] == 'victim')]['content'])

        hvv_combos += find_hvv_combinations(heroes, villains, victims)

    counter = Counter(hvv_combos)
    return counter.most_common(n)

def generate_combos(df, a, b):
    unique_art_ids = list(df['article_id'].unique())
    combos = []
    for art in unique_art_ids:
        all_a = list(df.loc[(df['article_id']==art) & (df['hvv']==a)]['clean_content'])
        all_b = list(df.loc[(df['article_id'] == art) & (df['hvv'] == b)]['clean_content'])
        for i in all_a:
            for j in all_b:
                combos.append(f"{i}</>{j}")
    return combos


def find_first_appears_in_partisanship_time(partisanship, start_time, duration, hvv):
    if ' ' in hvv:
        subset = df.loc[(df['partisanship'] == partisanship) & (df['time_index'] >= start_time) &
                                 (df['time_index'] <= start_time + duration)]
        part_words = generate_combos(subset,hvv.split(' ')[0], hvv.split(' ')[1])

    else:
        part_words = list(df.loc[(df['partisanship'] == partisanship) & (df['time_index'] >= start_time) &
                             (df['time_index'] <= start_time + duration) & (df['hvv']==hvv)]['clean_content'].unique())

    other_words = []
    for p in [x for x in partisanships if x != partisanship]:
        if ' ' in hvv:
            other_subset = df.loc[(df['partisanship'] == p) &
                                   (df['time_index'] <= start_time + duration)]
            other_words += generate_combos(other_subset,hvv.split(' ')[0], hvv.split(' ')[1] )
        else:
            other_words += list(df.loc[(df['partisanship'] == p) &
                                   (df['time_index'] <= start_time + duration) & (df['hvv']==hvv)]['clean_content'].unique())

    unique_to_part = [w for w in part_words if w not in other_words]

    high_occurring = []
    for kw in unique_to_part:
        if ' ' in hvv:
            freq = len([x for x in other_words if x==kw])
        else:
            freq = len(df.loc[(df['clean_content']==kw)])
        if freq >2:
            if kw not in high_occurring:
                high_occurring.append(kw)

    return high_occurring

def plot_word_over_time(word, partisanship, hvv=''):
    x, y = [], []
    for i in range(1, num_bins + 1):
        x.append(i)
        if "</>" in word:
            a = word.split("</>")[0]
            b = word.split("</>")[1]
            match_to_a = list(df.loc[(df['clean_content'] == a) & (df['partisanship'] == partisanship) &
                                 (df['time_index'] == i) & (df['hvv']==hvv.split(' ')[0])]['article_id'])
            count = 0
            for art_id in match_to_a:
                match_to_b = df.loc[(df['article_id']==art_id) & (df['clean_content'] == b)
                                    & (df['hvv']==hvv.split(' ')[1])]
                if len(match_to_b)>0:
                    count +=1
            y.append(count)

        else:
            y.append(len(df.loc[(df['clean_content'] == word) & (df['partisanship'] == partisanship) &
                                 (df['time_index'] == i)]))
    plt.plot(x, y, label=partisanship)



# clean_data() # Only needs to be run once
df = pd.read_csv('subsample_cleaned_hvv_data.csv')
df['time'] = pd.to_datetime(df['time'], format='ISO8601')
df['clean_content'] = df.apply(lambda x: remove_punct(x['content'].lower()) if isinstance(x['content'],str) else None, axis=1)

df['time_index'] = None
start = datetime.datetime(year=2016, month=1, day=1)
for i in range(1,num_bins+1):

    end = start + datetime.timedelta(days=15)
    df.loc[(df['time'] >= start) & (df['time'] < end), 'time_index'] = i
    start = end

# for p in partisanships:
#     find_most_common_triples(p, 5)
#
# for p in partisanships:
#     for hvv in  ['hero','villain','victim','all']:
#         mc = find_most_common_per_partisanship(hvv, p, df, 5)
#         print('')
#
# for yr in range(2016,2023):
#     for m in range(1,13):
#         for hvv in ['hero', 'villain', 'victim','all']:
#             mc = find_most_common_per_time(hvv, m,yr, df, 5)
#             print('')


already = []
for i in range(1,num_bins+1):
    duration = 1
    outcome = find_first_appears_in_partisanship_time('FarRight',i,duration,'hero villain')
    print(f"{i}-{i+duration}: {outcome}\n")

    for w in outcome:
        if w in already:
            continue
        already.append(w)
        # Plot appearances in all partisanhips and times
        for p in partisanships:
            plot_word_over_time(w, p, 'hero villain')

        plt.legend()
        plt.title(f"{w}")
        plt.show()


