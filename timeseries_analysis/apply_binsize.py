import shared_functions as sf
import pandas as pd
import random

# def extract_values(dictionary):
#     return dictionary['$oid']
#
# BIN_SIZE = 300
# pool_alphabet_files = sf.get_files_from_folder('pooled_alphabetized2', 'json')
#
# counter = {k:{j:{l: [] for l in range(1,13)} for j in range(2016,2023)} for k in ['FarRight', 'Right','CenterRight','Center','CenterLeft','Left','FarLeft']}
# content = []
# for file in pool_alphabet_files:
#     data = sf.import_json(file)['content']
#
    # df = pd.DataFrame(data)
    # df[3] = df[3].apply(lambda x: pd.Series(extract_values(x)))
    # df = df.drop_duplicates()
    # for index, row in df.iterrows():
    #     part = row.iloc[4]
    #     datetime_obj = pd.to_datetime(row.iloc[5])
    #     if datetime_obj is None:
    #         continue
#         year = datetime_obj.year
#         month = datetime_obj.month
#         counter[part][year][month].append(row.values.tolist())
#
#
# for p in counter.keys():
#     for yr in counter[p].keys():
#         for m in counter[p][yr].keys():
#             if len(counter[p][yr][m])>BIN_SIZE:
#                 sample = random.sample(counter[p][yr][m], BIN_SIZE)
#                 counter[p][yr][m] = sample
# print('')
#
# for p in counter.keys():
#     sf.export_as_json(f"sampled_pooled_alphabetized2/{p}_data.json",counter[p])
#
#

def flatten_file(loc):
    data = sf.import_json(loc)
    combo = []
    for year in data.keys():
        for m in data[year].keys():
            combo+=data[year][m]
    return combo


## Flatten the data
files = sf.get_files_from_folder('sampled_pooled_alphabetized2','json')
for file in files:
    flattened = flatten_file(file)
    sf.export_as_json(file,flattened)
