import matplotlib.pyplot as plt
import shared_functions as sf
import scipy.stats
import numpy as np

def fetch_correlation(level,h_v_v, origin, bin_size):
    if level == 'tuple':
        h_v_v = "-".join(h_v_v)
    # Fetch origin file
    if bin_size=='month':
        output = sf.import_json(f'../month_signals/{origin}_signals_by_part_input_hvv.json')[origin]
    else:
        output = sf.import_json(f'../signals/{origin}_signals_by_part_input_hvv.json')[origin]

    avg_correlation = {}
    for p_b in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
        if p_b==origin:
            continue

        rel_signals = output[p_b][level][h_v_v]
        corr = {k: [] for k in range(20)}
        for signal in rel_signals:
            r, p = scipy.stats.pearsonr(signal[1][:-1], signal[2][1:])  # coefficient, p-value
            if r <0.5:
                continue
            for l in range(20):
                if l == 0:
                    r, p = scipy.stats.pearsonr(signal[1], signal[2])  # coefficient, p-value
                else:
                    r, p = scipy.stats.pearsonr(signal[1][:-l], signal[2][l:])  # coefficient, p-value
                try:
                    x = int(r)
                    corr[l].append(r)
                except ValueError:
                    corr[l].append(None)

        avg_correlation[p_b] = list(corr.values())
    return avg_correlation

def plot_avg_correlation(level,h_v_v, origin,binsize):
    correlation = fetch_correlation(level,h_v_v,origin,binsize)
    x = list(range(20))
    for k in correlation.keys():
        for i in range(len(correlation[k])):
            try:

                plt.plot(x,[row[i] for row in correlation[k]])
            except IndexError:
                continue
    plt.xlabel('Time Shift')
    plt.ylabel('Average Correlation')
    plt.title(f'Correlation from Source: {origin}, for {h_v_v}')
    # plt.legend()

    # plt.savefig(f"..\\avg_corr_jamboard_viz\\{origin}_{h_v_v}_as_{level}_{binsize}.jpg")
    plt.show()


inputs = {
    # 'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}
for input_level in inputs.keys():
    for hvv in inputs[input_level]:
        for source in ['FarRight','Right','CenterRight','Center','CenterLeft','Left','FarLeft']:
            plot_avg_correlation(input_level, hvv, source,binsize='month')