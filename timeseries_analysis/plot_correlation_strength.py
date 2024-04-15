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


def fr_corr_strngth():

    other_data = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\FarRight_signals_by_part_input_hvv.json')
    data = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\signals_by_part_input_hvv.json')
    print('')

    source = data['FarRight']

    plottable = [['InputType','Partisanship','0.25>=x>0.5','0.5>=x>0.75','0.75>=']]
    for p in source.keys():
        if "Center" in p: # Ignoring the different types of Centrist
            for input_type in source[p].keys():
                corr_strength = breakdown_correlation_strength(source[p][input_type])
                plottable.append([input_type,p]+corr_strength)

    df = pd.DataFrame(data=plottable[1:],columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['0.25>=x>0.5','0.5>=x>0.75','0.75>=']).reset_index()
    ax = merged.plot.bar(x='Partisanship')
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.tight_layout()
    plt.title('Number of Centrist Narratives with a >0.25 Cross-Correlation with a Far Right Narrative by Correlation Strength')
    plt.savefig('graphs/num_positive_centristi_narr.png')
    plt.show()


def fl_corr_strngth():
    data = sf.import_json(
        'C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\FarLeft_signals_by_part_input_hvv.json')
    print('')

    source = data['FarLeft']

    plottable = [['InputType', 'Partisanship', '0.25>=x>0.5', '0.5>=x>0.75', '0.75>=']]
    for p in source.keys():
        if "Center" in p:  # Ignoring the different types of Centrist
            for input_type in source[p].keys():
                corr_strength = breakdown_correlation_strength(source[p][input_type])
                plottable.append([input_type, p] + corr_strength)

    df = pd.DataFrame(data=plottable[1:], columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['0.25>=x>0.5', '0.5>=x>0.75', '0.75>=']).reset_index()
    ax = merged.plot.bar(x='Partisanship')
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.tight_layout()
    plt.title(
        'Number of Centrist Narratives with a >0.25 Cross-Correlation with a Far Left Narrative by Correlation Strength')
    plt.savefig('graphs/FL_num_positive_centristi_narr.png')
    plt.show()

fl_corr_strngth()