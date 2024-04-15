"""

Compare the effect of the Far Right on Center partisanships vs partisan partisanships
"""
import shared_functions as sf
import pandas as pd
files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity','csv')]

data = [['input_level','hvv','part_a','part_b','count']]
for file in files:
    file_data = sf.import_csv(file)
    key_data = file.split('quantity\\')[1].replace('.csv','').split('_')
    if len(key_data)>3:
        data.append([key_data[1], key_data[2], key_data[0], key_data[3], len(file_data)])
    elif len(key_data)<=3 and 'Center' in key_data[2]:
        data.append([key_data[0], key_data[1], 'FarRight', key_data[2], len(file_data)])


df = pd.DataFrame(data=data[1:],columns=data[0])
x = df.groupby(by=['part_a','input_level']).sum('count')
df.to_csv('comprae_diff_part_influence.csv')

# COuld do chi square to show the difference
