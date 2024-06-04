"""
4/15 Rerunning analysis!

"""
from itertools import product
import sys
import os
import shared_functions as sf
import pandas as pd
import datetime
from collections import Counter
import scipy.stats
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def find_empty_times(times):
    # Given all the time data, look through to find any dates that are missing, and export them in a lis t
    new_values = []
    for yr, m, half in product(range(2016, 2023), range(1, 13), [1, 15]):
        if datetime.date(year=yr, month=m, day=half) not in times:
            new_values.append(datetime.date(year=yr, month=m, day=half))
    return new_values


def apply_binning(elts: list) -> list:
    # Given a list of data points, caputre the date info and put them into the appropriate bins 
    binned = []
    for elt in elts:
        date = elt[5]
        if date == 'ERROR: No publish date':
            continue
        datetime_obj = pd.to_datetime(date)
        if datetime_obj is None:
            continue

        month = datetime_obj.month
        year = datetime_obj.year

        if datetime_obj.day <= 15:
            month_half = 1
        else:
            month_half = 15
        doc = {'hero': elt[0],
               'villain': elt[1],
               'victim': elt[2],
               'combo': f"{elt[0]}.{elt[1]}.{elt[2]}",
               'month': month,
               'year': year,
               'month_half': month_half,
               'time': datetime.date(year=year, month=month, day=month_half),
               'partisanship': elt[4],
               'media_name': elt[6] if len(elt) > 6 else None}
        binned.append(doc)

    return binned


def fetch_all_characters(partisan_a:str, partisan_b:str, input_level:str, h_v_v:str) -> list:
    # Set the number of duplicates that must appear for the character to be counted
    threshold = 5

    partisan_files = [x for x in sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json') if f'\\{partisan_a}_'
                      in x or f"\\{partisan_b}" in x]
    hvv_indexing = {'hero': 0, 'villain': 1, 'victim': 2}

    content = []
    for file in partisan_files:
        data = sf.import_json(file)
        for elt in data:
            if elt[4] == partisan_a or elt[4] == partisan_b:
                if input_level == 'single':
                    content.append(elt[hvv_indexing[h_v_v]].lower())
                elif input_level == 'combo':
                    content.append(f"{elt[0]}.{elt[1]}.{elt[2]}".lower())
                # TODO: this gets either part a or part b, not the intersection! which means i'll have to be able 
                # to handle some empty intersections in the next function 
                elif input_level == 'tuple':
                    content.append(f"{elt[hvv_indexing[h_v_v[0]]]}.{elt[hvv_indexing[h_v_v[1]]]}".lower())
    counter = Counter(content).most_common()
    characters = [x[0] for x in counter if x[1] >= threshold and x[0] != 'none' and x[0] != 'time']
    return characters


def fetch_signal(character: str, hvv: str, partisanships, input_level) -> list:
    pool_alphabet_files = sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json')
    hvv_indexing = {'hero': 0, 'villain': 1, 'victim': 2}

    content = []
    for file in pool_alphabet_files:
        data = sf.import_json(file)
        if input_level == 'single':
            content += [x for x in data if x[hvv_indexing[hvv]] == character and x[4] in partisanships]
        elif input_level == 'combo':
            content += [x for x in data if f"{x[0]}.{x[1]}.{x[2]}".lower() == character and x[4] in partisanships]
        elif input_level == 'tuple':
            content += [x for x in data if
                        f"{x[hvv_indexing[hvv[0]]]}.{x[hvv_indexing[hvv[1]]]}".lower() == character and x[
                            4] in partisanships]

    # Convert to signal 
    binned_objs = apply_binning(content)
    df = pd.DataFrame(binned_objs)  # reformat

    df = df.drop_duplicates()  # remove any dupliactes, ie any time/partisanship combinations where the same
    # outlet appears more than once
    if len(df)==0:
        return [character, [],[]]

    freq_df = df[['time', 'partisanship']].value_counts().to_frame()

    freq_df.columns = [character]
    freq_df = freq_df.rename_axis(['time', 'partisanship']).reset_index()

    # For each partisanship, insert the missing dates
    for part in partisanships:
        empty_times = find_empty_times(freq_df.loc[freq_df.partisanship == part]['time'].to_list())
        for e in empty_times:
            freq_df = freq_df._append({character: 0,
                                       "time": e,
                                       "partisanship": part},
                                      ignore_index=True)

    freq_df = freq_df.sort_values(by='time')
    part_a_signal = [int(x) for x in freq_df[freq_df['partisanship'] == partisanships[0]][character].values]
    part_b_signal = [int(x) for x in freq_df[freq_df['partisanship'] == partisanships[1]][character].values]
    return [character, part_a_signal, part_b_signal]


def fetch_all_signals(data, h_v_v, partisanships, input_level):
    print(f"Fetching all {len(data)} signals")

    try: # look for existing file
        output = sf.import_json(f'signals/{partisanships[0]}_signals_by_part_input_hvv.json')
    except FileNotFoundError: # else create new one
        output = {partisanships[0]: {}}

    # Convert h_v_v into smth usable for dict key
    if isinstance(h_v_v, list):
        str_hvv = "-".join(h_v_v)
    else:
        str_hvv = str(h_v_v)

    # Prep container for signals
    if partisanships[0] in output.keys():
        if partisanships[1] in output[partisanships[0]].keys():
            if input_level in output[partisanships[0]][partisanships[1]].keys():
                if str_hvv in output[partisanships[0]][partisanships[1]][input_level].keys():
                    # if we already have the signal data, return it
                    return output[partisanships[0]][partisanships[1]][input_level][str_hvv]
                else:
                    output[partisanships[0]][partisanships[1]][input_level][str_hvv] = []
            else:
                output[partisanships[0]][partisanships[1]][input_level] = {str_hvv: []}
        else:
            output[partisanships[0]][partisanships[1]] = {input_level: {str_hvv: []}}
    else:
        output[partisanships[0]] = {partisanships[1]: {input_level: {str_hvv: []}}}

    # Fetch signals
    signals = [fetch_signal(c, h_v_v, partisanships, input_level) for c in data]

    output[partisanships[0]][partisanships[1]][input_level][str_hvv] = signals
    # output ={partisanships[0]: {partisanships[1]:{input_level:{h_v_v:signals}}}}

    filename = f'signals/{partisanships[0]}_signals_by_part_input_hvv.json'

    sf.export_as_json(filename, output)
    return signals


def find_fr_influence_narratives(signals):
    # Set the threshold for correlation strength
    threshold = 0.5

    high_correlation = []
    for signal_pair in signals:
        if len(signal_pair[1])<2 or len(signal_pair[2])<2:
            continue
        # Generate correlation 
        x, y = [], []
        for l in [1, 2]:
            r, p = scipy.stats.pearsonr(signal_pair[1][:-l], signal_pair[2][l:])  # coefficient, p-value
            x.append(l)
            y.append(r)
        if y[0] >= threshold or y[1] >= threshold:
            high_correlation.append([signal_pair[0], y])
    return high_correlation


# TODO: add other tuple combos 
inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}
for part_a in ['CenterLeft', 'Center', 'CenterRight','Left','FarLeft','Right','FarRight']:

    for input_level in inputs.keys():
        for hvv in inputs[input_level]:
            for part_b in ['CenterLeft', 'Center', 'CenterRight','Left','FarLeft','Right','FarRight']:
                print(f"For {input_level}, {hvv}, {part_b}:")
                characters = fetch_all_characters(part_a, part_b, h_v_v=hvv, input_level=input_level)
                signals = fetch_all_signals(characters, hvv, [part_a, part_b], input_level)
                high_correlation = find_fr_influence_narratives(signals)
                print(f"   {len(high_correlation)} narratives")
                if input_level == 'tuple':
                    sf.export_nested_list(f"cleaned_data_end_april\\{part_a}_{input_level}_{'-'.join(hvv)}_{part_b}.csv", high_correlation)
                else:
                    sf.export_nested_list(f"cleaned_data_end_april\\{part_a}_{input_level}_{hvv}_{part_b}.csv", high_correlation)

# TODO: add other part_a options, so that we can see the effect other parts have on
# the Center
