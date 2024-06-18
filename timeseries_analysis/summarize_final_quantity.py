import shared_functions as sf
import pandas as pd

def load_final_quantity_fr():
    # files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\final_quantity','csv') if '\\FarLeft' not in x and "\\Right" not in x
    #          and "\\Left" not in x]
    files =[x for x in sf.get_files_from_folder(
        'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_monthbins',
        'csv') if '\\FarRight' in x]
    data = [['input_level','hvv','part_b','character']]
    for file in files:
        # if 'Center' not in file:
        #     continue
        key_data = file.split('bins\\')[1].replace('.csv', '').split('_')
        input_level = key_data[1]
        hvv = key_data[2]
        part_b = key_data[3]
        file_data = sf.import_csv(file)
        for row in file_data:

            data.append([input_level, hvv, part_b, row[0]])
    df = pd.DataFrame(data=data[1:], columns=data[0])
    df.to_csv('tables/Final_Quantity_summary_monthbins.csv')

def load_final_quantity_fr_relative():
    complete_files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_monthbins','csv') if '\\FarLeft' not in x and "\\Right" not in x
                and "\\Left" not in x]
    # complete_files = [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april','csv') if '\\FarLeft' not in x and "\\Right" not in x
    #             and "\\Left" not in x]
    # partial_files =  [x for x in sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_segments','csv') if '\\FarLeft' not in x and "\\Right" not in x
    #             and "\\Left" not in x]
    signal  = sf.import_json('month_signals\\FarRight_signals_by_part_input_hvv.json')
    data = [['input_level','hvv','part_b','count','total_narr','time_frame (months)']]
    for file in complete_files:
        # if 'Center' not in file:
        #     continue
        file_data = sf.import_csv(file)
        key_data = file.split('end_april_monthbins\\')[1].replace('.csv','').split('_')
        input_level = key_data[1]
        hvv = key_data[2]
        part_b = key_data[3]
        num_narratives = len(signal['FarRight'][part_b][input_level][hvv])
        data.append([input_level, hvv, part_b, len(file_data), num_narratives, 84])
    # for file in partial_files:
    #     file_data = sf.import_csv(file)
    #     file_data = set([x[0] for x in file_data])
    #     key_data = file.split('end_april_segments\\')[1].replace('.csv', '').split('_')
    #     input_level = key_data[1]
    #     hvv = key_data[2]
    #     part_b = key_data[3]
    #     try:
    #         num_narratives = len(signal['FarRight'][part_b][input_level][hvv])
    #     except KeyError:
    #         continue
    #     time_frame = int(key_data[-1])/2
    #     data.append([input_level, hvv, part_b, len(file_data), num_narratives, time_frame])

    df = pd.DataFrame(data=data[1:], columns=data[0])
    df['relative'] = df['count']/df['total_narr']
    df.to_csv('tables/Final_Quantity_summary_relative_monthbins.csv')

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
# load_final_quantity_fl()
load_final_quantity_fr()