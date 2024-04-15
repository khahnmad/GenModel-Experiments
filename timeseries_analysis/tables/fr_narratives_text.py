import shared_functions as sf
import pandas as pd
files = sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity','csv')

data = [['input_level','hvv','part_b','narrative']]
for file in files:
    file_data = sf.import_csv(file)
    key_data = file.split('quantity\\')[1].replace('.csv','').split('_')
    if len(key_data)>3:
        continue
    for row in file_data:
        data.append([key_data[0], key_data[1], key_data[2], row[0]])
df = pd.DataFrame(data=data[1:],columns=data[0])
df.to_csv('Final_quantity_narratives.csv')