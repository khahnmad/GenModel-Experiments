"""
TODO: Run with complete FR final quantity narratives
"""
import shared_functions as sf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def breakdown_correlation_strength(data):
    a, b, c, d =0,0,0,0
    for hvv_type in data.keys(): # Not distinguishing between villain/ victim etc
        for c_signal_pair in data[hvv_type]:
            y = []
            for l in [1, 2]:
                r, p = scipy.stats.pearsonr(c_signal_pair[1][:-l], c_signal_pair[2][l:])  # coefficient, p-value
                y.append(r)
            if y[0] >= 0.75 or y[1] >= 0.75:
                d+=1
            elif y[0] >= 0.5 or y[1] >= 0.5:
                c+=1
            elif y[0] >= 0.25 or y[1] >= 0.25:
                b+=1
            elif y[0] >= 0 or y[1] >= 0:
                a+=1
    return [a,b,c,d]

def plot_nonfr_influence_on_center(binsize):
    plottable = [['Partisanship','0>=x>0.25','0.25>=x>0.5','0.5>=x>0.75','0.75>=']]

    if binsize=='month':
        files = sf.get_files_from_folder('..\\month_signals', 'json')
    else:
        files = sf.get_files_from_folder('..\\signals', 'json')
    for file in files:
        if 'signals\\FarRight' in file:
            continue
        data = sf.import_json(file)
        part_a = list(data.keys())[0]
        source = data[part_a]
        for p in source.keys():
            if 'Center' not in p:
                continue
            for input_type in source[p].keys():
                corr_strength = breakdown_correlation_strength(source[p][input_type])
                plottable.append([part_a]+corr_strength)

    df = pd.DataFrame(data=plottable[1:],columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['0>=x>0.25','0.25>=x>0.5','0.5>=x>0.75','0.75>=']).reset_index()
    # merged.plot.bar(x='Partisanship', stacked=True)
    ax=merged.set_index('Partisanship').loc[['FarLeft','Left','Right','FarRight']].plot.bar()
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.ylabel('# of Narratives')
    plt.tight_layout()
    plt.title('Narrative Influence on Centrist Partisanships by Correlation Strength')
    plt.savefig(f'other_influence_cr_parts_{binsize}')
    plt.show()

plot_nonfr_influence_on_center(binsize='month')