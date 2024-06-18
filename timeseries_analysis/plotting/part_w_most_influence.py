"""
- Bar plot, xaxis partisanship, y axis, number of matched narratives, grouped by Far Right origin vs far left origin
    - one plot per hvv/ input level

"""
import shared_functions as sf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

def breakdown_correlation_strength_by_hvv(data,hvv):
    count = 0
    # for hvv_type in data.keys(): # Not distinguishing between villain/ victim etc
    for c_signal_pair in data[hvv]:

        r, p = scipy.stats.pearsonr(c_signal_pair[1][:-1], c_signal_pair[2][1:])  # coefficient, p-value
        y = r
        if y>= 0.5 :
            count+=1

    return count

def breakdown_correlation_strength(data):
    count = 0
    for hvv_type in data.keys(): # Not distinguishing between villain/ victim etc
        for c_signal_pair in data[hvv_type]:

            r, p = scipy.stats.pearsonr(c_signal_pair[1][:-1], c_signal_pair[2][1:])  # coefficient, p-value
            y = r
            if y>= 0.5 :
                count+=1

    return count


def plot_total_count():
    fr_data = sf.import_json('..\\month_signals\\FarRight_signals_by_part_input_hvv.json')['FarRight']
    fl_data =  sf.import_json('..\\month_signals\\FarLeft_signals_by_part_input_hvv.json')['FarLeft']


    plottable = [['Partisanship','FR Count','FL Count']]
    for p in ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right','FarRight']:
        for input_type in fr_data[p].keys():
            fr_count = breakdown_correlation_strength(fr_data[p][input_type])
            fl_count = breakdown_correlation_strength(fl_data[p][input_type])
            plottable.append([p, fr_count, fl_count])

    df = pd.DataFrame(data=plottable[1:], columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['FR Count','FL Count']).reset_index()
    ax = merged.set_index('Partisanship').loc[
        ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right','FarRight']].plot.bar()
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.ylabel('# of Narratives')
    plt.tight_layout()
    plt.title('Comparing Far Right to Far Left influence on all partisanships')
    # plt.savefig('fr_influence_other_parts')
    plt.show()

def plot_by_input_level(input_type, hvv_type):
    fr_data = sf.import_json('..\\month_signals\\FarRight_signals_by_part_input_hvv.json')['FarRight']
    fl_data = sf.import_json('..\\month_signals\\FarLeft_signals_by_part_input_hvv.json')['FarLeft']

    plottable = [['Partisanship', 'FR Count', 'FL Count']]
    for p in ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right','FarRight']:
        # for input_type in fr_data[p].keys():
        fr_count = breakdown_correlation_strength_by_hvv(fr_data[p][input_type], hvv_type)
        fl_count = breakdown_correlation_strength_by_hvv(fl_data[p][input_type], hvv_type)
        plottable.append([p, fr_count, fl_count])

    df = pd.DataFrame(data=plottable[1:], columns=plottable[0])
    merged = df.groupby(by='Partisanship').sum(['FR Count', 'FL Count']).reset_index()
    ax = merged.set_index('Partisanship').loc[
        ['FarLeft', 'Left', 'CenterLeft', 'Center', 'CenterRight', 'Right','FarRight']].plot.bar()
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend()
    plt.ylabel('# of Narratives')
    plt.tight_layout()
    plt.title(f'{hvv_type}, {input_type}: Comparing Far Right to Far Left influence on all partisanships')
    # plt.savefig('fr_influence_other_parts')
    plt.show()

plot_total_count()

inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[ 'hero-villain','hero-victim','villain-victim']
}
for input_type in inputs.keys():
    for hvv_type in inputs[input_type]:
        plot_by_input_level(input_type, hvv_type)