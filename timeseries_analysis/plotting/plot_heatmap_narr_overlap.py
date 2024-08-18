import matplotlib.pyplot as plt
import shared_functions as sf
import pandas as pd

def generate_grid(metric):
    # prepare data for full narrative in common
    files = [x for x in sf.get_files_from_folder(
        r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\sampled_pooled_alphabetized2',
        'json')]
    records = {part:[] for part in sf.PARTISANSHIPS}
    for file in files:
        partisanship = file.split('alphabetized2')[1].split('_data.json')[0].replace('\\','')
        data = sf.import_json(file)
        narratives = [x[:3] for x in data]
        records[partisanship] = narratives

    formatted = []
    for p_a in records.keys():
        new = {'part_x':p_a}
        for p_b in records.keys():
            if p_a==p_b:
                new[p_b]= None
                continue
            if metric=='characters':
                char_a, char_b = [".".join(sorted(x[:3])) for x in records[p_a]], [".".join(sorted(x[:3])) for x in records[p_b]]
                new[p_b] = len(set(char_a).intersection(set(char_b)))
            elif metric=='narratives':
                narr_a, narr_b = [".".join(x[:3]) for x in records[p_a]], [".".join(x[:3]) for x in records[p_b]]
                new[p_b] = len(set(narr_a).intersection(set(narr_b)))

        formatted.append(new)
    df = pd.DataFrame(formatted)
    return df

def plot_heatmap(metric):
    data = generate_grid(metric)

    parts =[x for x in data.columns if x!='part_x']
    grid = data[parts].values
    plt.imshow(grid)
    plt.colorbar()

    plt.title(f'Overlap by Partisanship for {metric}')
    plt.xticks(range(len(grid)),
               parts, rotation=90)
    plt.ylabel('Partisanship')
    plt.yticks(range(len(grid)),
               parts)
    plt.xlabel('Partisanship')
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f"..\\heatmap_jamboard_viz\\{he_vi_vic}_as_{level}_{segment_length}_{binsize}.jpg")
    plt.show()
inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}


plot_heatmap('characters')
plot_heatmap('narratives')
# data = np.random.random((12, 12))
