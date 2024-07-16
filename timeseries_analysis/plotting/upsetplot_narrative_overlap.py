from upsetplot import generate_counts
from upsetplot import plot

import shared_functions as sf
import matplotlib.pyplot as plt
import pandas as pd


# starting with the full narrtive combination
def fetch_narratives(partisan):
    partisan_files = [x for x in sf.get_files_from_folder('../sampled_pooled_alphabetized2', 'json') if f'\\{partisan}_'
                      in x]
    hvv_indexing = {'hero': 0, 'villain': 1, 'victim': 2}

    content = []
    for file in partisan_files:
        data = sf.import_json(file)
        for elt in data:
            if elt[4] == partisan :
                # if input_level == 'single':
                #     content.append(elt[hvv_indexing[h_v_v]].lower())
                # elif input_level == 'combo':
                narrative = f"{elt[0]}.{elt[1]}.{elt[2]}".lower()
                if narrative=='none.none.none':
                    continue
                if narrative not in content:
                    content.append(narrative)
                # elif input_level == 'tuple':
                #     content.append(f"{elt[hvv_indexing[h_v_v[0]]]}.{elt[hvv_indexing[h_v_v[1]]]}".lower())
    return content

def count_overlap(data_a, data_b):
    return len(set(data_a).intersection(set(data_b))) / min(len(data_a), len(data_b))


# example = generate_counts()
#
# plot(example)
# plt.show()

part_data = {}
for part in sf.PARTISANSHIPS:
    part_data[part] = fetch_narratives(part)

records = []
for p_a in part_data.keys():
    for p_b in part_data.keys():
        if p_a==p_b:
            continue
        record = {}
        for part in sf.PARTISANSHIPS:
            if part==p_a or part==p_b:
                record[part] = True
            else:
                record[part]= False
        record['value'] = count_overlap(part_data[p_a],part_data[p_b])
        records.append(record)

df = pd.DataFrame(records)
df = df.drop_duplicates()
as_series = df.set_index(sf.PARTISANSHIPS)['value']
# plot(as_series, sort_by='degree')
# plot(as_series, sort_by='cardinality')
plot(as_series, sort_categories_by='input',sort_by='cardinality')
plt.show()

conversion = {'Right':['FarRight',"Right"],
              "Center":['Center','CenterLeft',"CenterRight"],
              'Left':['Left','FarLeft']}
# Blur into partisan extremes, Right, Center, Left
records = []
for p_a in ['Right', "Center","Left"]:
    for p_b in ['Right', "Center","Left"]:
        if p_a==p_b:
            continue
        record = {}
        for part in ['Right', "Center","Left"]:
            if part==p_a or part==p_b:
                record[part] = True
            else:
                record[part]= False
        part_a_data = []
        for k in conversion[p_a]:
            part_a_data+=part_data[k]
        part_b_data = []
        for k in conversion[p_b]:
            part_b_data += part_data[k]
        record['value'] = count_overlap(part_a_data,part_b_data)
        records.append(record)

df = pd.DataFrame(records)
df = df.drop_duplicates()
as_series = df.set_index(['Right','Center','Left'])['value']
# plot(as_series, sort_by='degree')
# plot(as_series, sort_by='cardinality')
plot(as_series, sort_categories_by='input',sort_by='cardinality')
plt.show()