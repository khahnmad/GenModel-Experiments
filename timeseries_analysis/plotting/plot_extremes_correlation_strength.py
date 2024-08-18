import shared_functions as sf
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def calculate_correlation(year, signals):
    year_translation = {2016: (0, 12),
                        2017: (12, 24),
                        2018: (24, 36),
                        2019: (36, 48),
                        2020: (48, 60),
                        2021: (60, 72),
                        2022: (72, 84)
                        }

    correlation = []
    a, b = year_translation[year]
    for signal_pair in signals:
        if len(signal_pair[1][a:b]) < 2 or len(signal_pair[2][a:b]) < 2:
            continue
        # Generate correlation

        r, p = scipy.stats.pearsonr(signal_pair[1][a:b][:-1],
                                    signal_pair[2][a:b][1:])  # coefficient, p-value
        try:
            int(r)
            correlation.append(r)
        except ValueError:
            continue

    return correlation


def import_data(part_a, part_b):

    output = sf.import_json(f'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\month_signals\\{part_a}_signals_by_part_input_hvv.json')
    signals = output[part_a][part_b]['combo']['hero, villain, victim']

    return signals

def load_data():
    records = []
    for p_a in ['FarRight','FarLeft']:
        for p_b in  ['FarRight','FarLeft']:
            if p_a==p_b:
                continue
            signals = import_data(p_a, p_b)
            for year in range(2016,2023):
                corr = calculate_correlation(year, signals)
                for c in corr:
                    records.append({'direction':f"{p_a}->{p_b}",
                                    'year':year,
                                    'correlation':c})
    df = pd.DataFrame(records)
    return df

def plot_histograms():


    # Data
    apples_2016 = [0.2, 0.3, 0.4, 0.5]
    pears_2016 = [0.6, 0.7, 0.8, 0.9]
    apples_2017 = [0.2, 0.3, 0.4, 0.6]
    pears_2017 = [0.6, 0.7, 0.4, 0.9]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))


    axs[0, 0].hist(apples_2016, bins=4, alpha=0.5, label='Apples 2016', color='red')
    axs[0, 0].set_title('Apples 2016')

    axs[0, 1].hist(pears_2016, bins=4, alpha=0.5, label='Pears 2016', color='green')
    axs[0, 1].set_title('Pears 2016')

    axs[1, 0].hist(apples_2017, bins=4, alpha=0.5, label='Apples 2017', color='red')
    axs[1, 0].set_title('Apples 2017')

    axs[1, 1].hist(pears_2017, bins=4, alpha=0.5, label='Pears 2017', color='green')
    axs[1, 1].set_title('Pears 2017')

    for ax in axs.flat:
        ax.set(xlabel='Scores', ylabel='Frequency')

    plt.tight_layout()
    plt.show()

def plot_boxplot(data):
    # Data
    # data = pd.DataFrame({
    #     'Scores': apples_2016 + pears_2016 + apples_2017 + pears_2017,
    #     'Category': ['Apples'] * len(apples_2016) + ['Pears'] * len(pears_2016) + ['Apples'] * len(apples_2017) + [
    #         'Pears'] * len(pears_2017),
    #     'Year': ['2016'] * len(apples_2016) + ['2016'] * len(pears_2016) + ['2017'] * len(apples_2017) + ['2017'] * len(
    #         pears_2017)
    # })

    sns.boxplot(x='direction', y='correlation', hue='year', data=data)
    plt.title('Box Plot of Narrative Correlation by influencing Partisanship and Year')
    plt.show()


if __name__ == '__main__':
    df = load_data()

    plot_boxplot(df)

