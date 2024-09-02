"""
PLot % of narratives that match / narratives that overlap for FR -> all parts
x=FR vs. FL
y= % narr match / overlap
hue=partisanship b
"""
import shared_functions as sf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_num_matching(part_a,part_b):
    file_loc = f"../cleaned_data_end_april_monthbins/{part_a}_combo_hero, villain, victim_{part_b}.csv"
    data = sf.import_csv(file_loc)
    return len(data)

def get_overlapping_narratives(part_a, part_b):
    # prepare data for full narrative in common
    files = [x for x in sf.get_files_from_folder(
        r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\sampled_pooled_alphabetized2',
        'json') if  part_a in x or part_b in x]
    if len(files)<2:
        print('')
    narrs = []
    for file in files:
        data = sf.import_json(file)
        narratives = [x[:3] for x in data]
        unsorted_narratives = [(".".join(sorted(x))) for x in narratives]
        narrs.append(unsorted_narratives)

    overlap = set(narrs[0]).intersection(set(narrs[1]))
    return len(overlap)

records = []
for p_a in ['FarRight','FarLeft']:
    for p_b in sf.PARTISANSHIPS:
        if p_a==p_b:
            continue
        # get matching narratives
        num_matching = get_num_matching(p_a,p_b)
        # get overlapping narratives
        num_overlapping = get_overlapping_narratives(p_a, p_b)
        records.append({'Influencer':p_a,
                        'Influenced':p_b,
                        'Percent Matching':(num_matching/num_overlapping)*100})


df = pd.DataFrame(records)

sns.barplot(x='Influencer', y='Percent Matching', hue='Influenced', data=df)
plt.title('Bar Plot of the percent of narratives that match over narratives that overlap for Far Right and Far Left as Influencers')
plt.show()

