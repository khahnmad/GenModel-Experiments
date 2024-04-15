"""
Inputs: Metadata file
Outputs: string text of the narratives with the strongest fr influence by category
        - categories: single hvv, combo hvv, tuple hvv, h->v

"""
import shared_functions as sf
import pandas as pd

# def get_strongest_single_combo_narratives(hvv:str)->list:
#     data = sf.import_json(f"C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\imeseries_copies\\outlet\\15day\\{hvv}\\metadata_file.csv")
#     df = pd.DataFrame(data=data[1:], columns=data[0])
#     subset = df[(df['part_a'] == 'FarRight') & (df['part_b'] != 'FarRight')]
#     high_value = subset[(subset['lag=1'] >= 0.5) | (subset['lag=2'] >= 0.5) | (subset['lag=3'] >= 0.5)]
#     return list(high_value.values)



def get_strongest_fr_narratives(input_level:str,hvv:list):
    data = sf.import_csv(
        f"C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\timeseries_copies\\v2\\{input_level}\\{'_'.join(hvv)}\\metadata_file.csv")
    df = pd.DataFrame(data=data[1:], columns=data[0])
    for i in range(5):
        df[f'lag={i}'] = df[f'lag={i}'].astype(float)
    subset = df[(df['part_a'] == 'FarRight') & (df['part_b'] != 'FarRight')]
    high_value = subset[(subset['lag=1'] >= 0.65) | (subset['lag=2'] >= 0.65) | (subset['lag=3'] >= 0.65)]
    return list(high_value.values)

    # if input_level=='single':
    #     return get_strongest_single_combo_narratives(hvv[0])
    # if input_level=='tuple':
    #     return None
    # if input_level=='combo':
    #     return get_strongest_single_combo_narratives(input_level)
    # if input_level=='transition':
    #     return None

if __name__ == '__main__':
    input_levels = {'single':[['villain'], ['hero'],['victim']],
                    # 'tuple':[['hero', 'villain'], ['hero', 'victim'], ['villain', 'victim']],
                    'combo':[['hero','villain','victim']],
                    # 'transition':[['hero', 'villain'], ['hero', 'victim'], ['villain', 'hero'], ['villain', 'victim'],
                    #               ['victim', 'hero'], ['victim', 'villain']]
                  }
    for k in input_levels.keys():
        for hvv in input_levels[k]:
            filename = f'pooled/{k}_{"_".join(hvv)}_strongest_fr_narratives.csv'
            print(f"Running {k}: {'_'.join(hvv)}")
            narratives = get_strongest_fr_narratives(input_level=k, hvv=hvv)
            sf.export_nested_list(filename,narratives)
    files = sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\pooled','csv')
    for file in files:
        print(file)
        data = sf.import_csv(file)
        for row in data:
            print(row[2:10])
        print('\n\n')
