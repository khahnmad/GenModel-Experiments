import shared_functions as sf
import matplotlib.pyplot as plt

complete_files = [x for x in sf.get_files_from_folder(
    'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april', 'csv')]
partial_files = [x for x in sf.get_files_from_folder(
    'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_segments',
    'csv')]
signal = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\signals_by_part_input_hvv.json')

for input_level  in ['single','combo','tuple']:
    # x_diverse,y =[],[]
    xy = {}
    rel_files = [x for x in complete_files+partial_files if input_level in x]
    for file in rel_files:
        if 'end_april\\' in file:
            key_data = file.split('end_april\\')[1].replace('.csv', '').split('_')
            segment_length = 84
            partisanship = key_data[-1]
        else:
            key_data = file.split('segments\\')[1].replace('.csv','').split('_')
            segment_length = int(key_data[-1])
            partisanship = key_data[-2]
        if partisanship not in xy.keys():
            xy[partisanship] = [[],[]]
        data = sf.import_csv(file)
        xy[partisanship][0].append(segment_length)
        xy[partisanship][1].append(len(data))
    for k in xy.keys():
        plt.scatter(xy[k][0],xy[k][1], label=k)
    plt.legend()
    plt.xlabel('Segment Length')
    plt.ylabel('Number of Narratives')
    plt.title(f"{input_level}")
    plt.show()