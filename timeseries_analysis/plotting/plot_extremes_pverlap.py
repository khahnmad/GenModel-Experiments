import shared_functions as sf
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


def count_full_unique_narr_shared():
    # prepare data for full narrative in common
    partisan_files = [x for x in sf.get_files_from_folder(r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\sampled_pooled_alphabetized2', 'json') if f'\\FarRight_'
                          in x or f"\\FarLeft" in x]
    records = {yr: {"FarRight":[],'FarLeft':[]} for yr in range(2016, 2023)}
    for file in partisan_files:
        if 'FarRight' in file:
            partisanship = 'FarRight'
        else:
            partisanship = 'FarLeft'

        data = sf.import_json(file)
        for yr in range(2016,2023):
            narratives = [f"{x[0]}.{x[1]}.{x[2]}" for x in data if f'{yr}-' in x[5]]
            records[yr][partisanship] = narratives
    formatted = []
    for k in records.keys():
        new = {}
        for kk in records[k].keys():
            new[f'num_{kk}'] = len(set(records[k][kk])) # count the number of unique narratives
        new['shared'] = len(set(records[k]['FarRight']).intersection(set(records[k]['FarLeft'])))
        new['year'] = k
        formatted.append(new)
    df = pd.DataFrame(formatted)
    return df

def count_full_unique_char_shared():
    # prepare data for full narrative in common
    partisan_files = [x for x in sf.get_files_from_folder(r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\sampled_pooled_alphabetized2', 'json') if f'\\FarRight_'
                          in x or f"\\FarLeft" in x]
    records = {yr: {"FarRight":[],'FarLeft':[]} for yr in range(2016, 2023)}
    for file in partisan_files:
        if 'FarRight' in file:
            partisanship = 'FarRight'
        else:
            partisanship = 'FarLeft'

        data = sf.import_json(file)
        for yr in range(2016,2023):

            narratives = [".".join(sorted(x[:3])) for x in data if f'{yr}-' in x[5]]
            records[yr][partisanship] = narratives
    formatted = []
    for k in records.keys():
        new = {}
        for kk in records[k].keys():
            new[f'num_{kk}'] = len(set(records[k][kk])) # count the number of unique narratives
        new['shared'] = len(set(records[k]['FarRight']).intersection(set(records[k]['FarLeft'])))
        new['year'] = k
        formatted.append(new)
    df = pd.DataFrame(formatted)
    return df

def plot_bar_chart(df, title):
    # Data
    N = len(df['year'])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, df['num_FarRight'].values, width, label='FarRight')
    p2 = ax.bar(ind + width, df['num_FarLeft'].values, width, label='FarLeft')
    p3 = ax.bar(ind + width / 2, df['shared'].values, width / 2, label='Shared', color='grey')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel(f'Number of {title}')
    ax.set_title(f'{title} by year and overlap')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels((df['year'].values))
    ax.legend()

    plt.show()

def plot_stacked_bar_chart(df, title):
    # Data
    ind = np.arange(len(df['year']))  # the x locations for the groups
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    # Bottom bar - non-shared friends
    p1 = ax.bar(ind, [j - s for j, s in zip(df['num_FarRight'].values, df['shared'].values)], width, label='Only FarRight')
    p2 = ax.bar(ind, [s for s in df['shared'].values], width, bottom=[j - s for j, s in zip(df['num_FarRight'].values, df['shared'].values)],
                label='Shared friends', color='grey')
    p3 = ax.bar(ind, [j - s for j, s in zip(df['num_FarLeft'].values, df['shared'].values)], width, bottom=[j for j in df['num_FarRight'].values],
                label='Only Far Left')

    ax.set_ylabel(f'Number of {title}')
    ax.set_title(f'{title} by year and overlap')
    ax.set_xticks(ind)
    ax.set_xticklabels(df['year'])
    ax.legend()
    plt.show()

def plot_area_chart(df, title):


    # Data
    fig, ax = plt.subplots(1,2,figsize=(12,5))

    # Area
    ax[0].fill_between(df['year'], df['num_FarRight'], color='red', alpha=0.4, label='Far Right')
    ax[0].fill_between(df['year'], df['num_FarLeft'], color='blue', alpha=0.4, label='Far Left')
    ax[0].fill_between(df['year'], df['shared'], color='black', alpha=0.6, label='Shared')

    ax[0].set_ylabel(f'Number of {title}')
    ax[0].set_title(f'{title} by year and overlap')
    ax[0].legend()

    # just the line plot of shared narr over time
    ax[1].plot(df['year'].values,df['shared'].values)
    ax[1].set_ylabel(f'Number of Shared {title}')
    ax[1].set_title(f'{title} Shared Over Time')
    plt.show()

if __name__ == '__main__':
    # metric = 'Unique, Complete, Characters'
    metric = 'Unique Narratives'
    df = count_full_unique_narr_shared()
    # df = count_full_unique_char_shared()
    plot_area_chart(df,metric)
    plot_bar_chart(df,metric)
    plot_stacked_bar_chart(df, metric)