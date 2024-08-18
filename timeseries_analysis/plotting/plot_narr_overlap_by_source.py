import shared_functions as sf
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from upsetplot import UpSet



def calculate_overlap(fr_values, fl_values, metric):
    shared_heroes = len(set(fr_values).intersection(set(fl_values)))
    return {'FarRight': len(set(fr_values)), 'FarLeft': len(set(fl_values)), 'Shared': shared_heroes, 'Metric': metric}


def generate_overall_character_count():
    # prepare data for full narrative in common
    files = [x for x in sf.get_files_from_folder(
        r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\sampled_pooled_alphabetized2',
        'json') if 'FarLeft' in x or 'FarRight' in x]

    records = {}
    for file in files:
        partisanship = file.split('alphabetized2')[1].split('_data.json')[0].replace('\\','')
        data = sf.import_json(file)
        narratives = [x[:3] for x in data]
        records[partisanship] = narratives


    formatted = []
    hvv_conversion = {0:'hero',1:'villain',2:'victim'}
    for i in range(3):
        # heroes
        a_heroes, b_heroes =[x[i] for x in records['FarRight']], [x[i] for x in records['FarLeft']]
        formatted.append(calculate_overlap(a_heroes,b_heroes,hvv_conversion[i]))
    already_done = []
    for i in range(3):
        for j in range(3):
            if i==j:
                continue
            if "".join(sorted([str(i),str(j)])) in already_done:
                continue
            a_values, b_values = [f"{x[i]}.{x[j]}" for x in records['FarRight']], [f"{x[i]}.{x[j]}" for x in records['FarLeft']]
            formatted.append(calculate_overlap(a_values,b_values,f"{hvv_conversion[i]}-{hvv_conversion[j]}"))
            already_done.append("".join(sorted([str(i),str(j)])))
    a_values, b_values = [f".".join(x[:3]) for x in records['FarRight']], [f".".join(x[:3]) for x in records['FarLeft']]
    formatted.append(calculate_overlap(a_values, b_values, f"hero-villain-victim"))

    df = pd.DataFrame(formatted)
    return df


def plot_stacked_bar_chart(df):


    # Data
    sources = df['Metric'].values
    janes_friends = df['FarRight'].values
    jims_friends = df['FarLeft'].values
    shared_friends = df['Shared'].values

    ind = np.arange(len(sources))  # the x locations for the groups
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    # Bottom bar - non-shared friends
    p1 = ax.bar(ind, [j - s for j, s in zip(janes_friends, shared_friends)], width, label="FarRight Only")
    p2 = ax.bar(ind, shared_friends, width, bottom=[j - s for j, s in zip(janes_friends, shared_friends)],
                label='Shared', color='grey')
    p3 = ax.bar(ind, [j - s for j, s in zip(jims_friends, shared_friends)], width, bottom=[j for j in janes_friends],
                label="FarLeft Only")

    ax.set_ylabel('Number shared')
    ax.set_title('Shared by source and overlap')
    ax.set_xticks(ind)
    ax.set_xticklabels(sources)
    ax.legend()

    plt.show()


def plot_grouped_bar_chart(df):
    # Data
    sources = df['Metric'].values
    janes_friends = df['FarRight'].values
    jims_friends = df['FarLeft'].values
    shared_friends = df['Shared'].values

    ind = np.arange(len(sources))  # the x locations for the groups
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()

    # Plotting the bars
    p1 = ax.bar(ind, janes_friends, width, label="FarRight")
    p2 = ax.bar(ind + width, jims_friends, width, label="FarLeft")
    p3 = ax.bar(ind + width / 2, shared_friends, width / 2, label="Shared", color='grey')

    # Adding labels, title and legend
    ax.set_ylabel('Number shared')
    ax.set_title('Shared by source and overlap')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(sources)
    ax.legend()

    plt.show()


def plot_upset_plot(data):
    # Transforming the data into a format suitable for UpSet plot
    upset_data = {
        ("FarRight",): data["FarRight"] - data["Shared"],
        ("FarLeft",): data["FarLeft"] - data["Shared"],
        ("FarRight", "FarLeft"): data["Shared"],
    }

    upset_df = pd.DataFrame(upset_data)
    upset_df.index = data["Metric"].values

    # Plotting the UpSet plot
    upset = UpSet(upset_df, intersection_plot_elements=len(data['Metric']))
    upset.plot()
    plt.suptitle('Shared by source and overlap')
    plt.show()


if __name__ == '__main__':
    df = generate_overall_character_count()
    bar_Df = df[['Shared']].T
    bar_Df.columns = df['Metric']
    bar_Df.index = [0]
    bar_Df = bar_Df.T
    bar_Df.plot.bar()
    plt.title('Number of Shared Narratives & Narrative Segments between the Far Left and Far Right')
    plt.tight_layout()
    plt.show()

    # plot_upset_plot(df)
    # plot_stacked_bar_chart(df)
    # plot_grouped_bar_chart(df)