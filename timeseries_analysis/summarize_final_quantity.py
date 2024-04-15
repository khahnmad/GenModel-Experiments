import shared_functions as sf
import pandas as pd

def load_final_quantity_fr():
    files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity','csv') if '\\FarLeft' not in x and "\\Right" not in x
             and "\\Left" not in x]
    data = [['input_level','hvv','part_b','count']]
    for file in files:
        if 'Center' not in file:
            continue
        file_data = sf.import_csv(file)
        key_data = file.split('quantity\\')[1].replace('.csv','').split('_')
        input_level = key_data[0]
        hvv = key_data[1]
        part_b = key_data[2]
        data.append([input_level, hvv, part_b, len(file_data)])
    df = pd.DataFrame(data=data[1:], columns=data[0])
    df.to_csv('tables/Final_Quantity_summary.csv')

def load_final_quantity_fr_relative():
    files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity','csv') if '\\FarLeft' not in x and "\\Right" not in x
                and "\\Left" not in x]
    signal  = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\signals_by_part_input_hvv.json')
    data = [['input_level','hvv','part_b','count','total_narr']]
    for file in files:
        if 'Center' not in file:
            continue
        file_data = sf.import_csv(file)
        key_data = file.split('quantity\\')[1].replace('.csv','').split('_')
        input_level = key_data[0]
        hvv = key_data[1]
        part_b = key_data[2]
        num_narratives = len(signal['FarRight'][part_b][input_level][hvv])
        data.append([input_level, hvv, part_b, len(file_data), num_narratives])
    df = pd.DataFrame(data=data[1:], columns=data[0])
    df['relative'] = df['count']/df['total_narr']
    df.to_csv('tables/Final_Quantity_summary_relative.csv')

def load_final_quantity_fl():
    files = [x for x in
             sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity', 'csv') if
             '\\FarLeft' in x]
    data = [['input_level', 'hvv', 'part_b', 'narrative']]
    for file in files:
        if 'Center' not in file:
            continue
        file_data = sf.import_csv(file)
        key_data = file.split('quantity\\')[1].replace('.csv', '').split('_')
        input_level = key_data[1]
        hvv = key_data[2]
        part_b = key_data[3]
        for row in file_data:
            data.append([input_level, hvv, part_b, row[0]])
    df = pd.DataFrame(data=data[1:], columns=data[0])

    df.to_csv('tables/FL_final_narratives.csv')

load_final_quantity_fr_relative()