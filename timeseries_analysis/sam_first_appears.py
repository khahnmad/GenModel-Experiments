'C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals'
import shared_functions as sf

def get_first_appearance(values):
    for i in range(len(values)):
        if values[i]>0:
            return i
    return len(values)


data = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\signals_by_part_input_hvv.json')['FarRight']
first_appearances = {'FarRight':[]}
for p in data.keys():
    if p not in first_appearances.keys():
        first_appearances[p]=[]
    for input_level in data[p].keys():
        for hvv in data[p][input_level].keys():
            for c_signal_pair in data[p][input_level][hvv]:
                if 'tara reade' in c_signal_pair[0]:
                    first_appearances['FarRight'].append(get_first_appearance(c_signal_pair[1]))
                    first_appearances[p].append(get_first_appearance(c_signal_pair[2]))
