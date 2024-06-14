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
    return [b,c,d]


def plot_fr_influence(binsize):
    if binsize=='month':
        data = sf.import_json('..\\month_signals\\FarRight_signals_by_part_input_hvv.json')
    else:
        data = sf.import_json('..\\signals\\FarRight_signals_by_part_input_hvv.json')
    source = data['FarRight']

    plottable = [['Partisanship','0.25>=x>0.5','0.5>=x>0.75','0.75>=']]
    for p in ['FarLeft','Left','CenterLeft','Center','CenterRight','Right']:
        for input_type in source[p].keys():
            corr_strength = breakdown_correlation_strength(source[p][input_type])
            plottable.append([p]+corr_strength)

    df = pd.DataFrame(data=plottable[1:],columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['0.25>=x>0.5','0.5>=x>0.75','0.75>=']).reset_index()
    ax=merged.set_index('Partisanship').loc[['FarLeft','Left','CenterLeft','Center','CenterRight','Right']].plot.bar()
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.ylabel('# of Narratives')
    plt.tight_layout()
    plt.title('Far Right Narrative Influence on Other Partisanships by Correlation Strength')
    plt.savefig('fr_influence_other_parts')
    plt.show()

plot_fr_influence(binsize='month')