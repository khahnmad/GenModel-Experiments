import matplotlib.pyplot as plt
import shared_functions as sf

def generate_grid(i_level, hvv, length):
    grid = []
    partisanships = ['FarRight','Right','CenterRight', 'Center','CenterLeft','Left','FarLeft']
    for p_a in partisanships:
        row = []
        for p_b in partisanships:
            if i_level == 'tuple':
                if isinstance(length,int):
                    file = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_segments\\{p_a}_{i_level}_{'-'.join(hvv)}_{p_b}_{length}.csv"
                else:
                    file = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april\\{p_a}_{i_level}_{ '-'.join(hvv)}_{p_b}.csv"
            else:
                if isinstance(length,int):
                    file = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_segments\\{p_a}_{i_level}_{hvv}_{p_b}_{length}.csv"
                else:
                    file = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april\\{p_a}_{i_level}_{hvv}_{p_b}.csv"
            try:
                data = sf.import_csv(file)
            except FileNotFoundError:
                data = []
            row.append(len(data))
        grid.append(row)
    return grid

def plot_heatmap(level, he_vi_vic, segment_length):
    data = generate_grid(level, he_vi_vic, segment_length)
    parts = ['FarRight','Right','CenterRight', 'Center','CenterLeft','Left','FarLeft']
    plt.imshow(data)
    plt.colorbar()
    if level=='tuple':
        he_vi_vic="-".join(he_vi_vic)
    plt.title(f"{he_vi_vic} as {level}: {segment_length}")
    plt.xticks(range(len(data)),
               parts, rotation=90)
    plt.ylabel('Original Signal')
    plt.yticks(range(len(data)),
               parts)
    plt.xlabel('Delayed Signal')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"heatmap_jamboard_viz\\{he_vi_vic}_as_{level}_{segment_length}.jpg")
    plt.show()
inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}

for input_level in inputs.keys():
    for hvv in inputs[input_level]:
        plot_heatmap(input_level, hvv, segment_length=20)
# data = np.random.random((12, 12))
