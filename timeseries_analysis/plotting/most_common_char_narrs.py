import matplotlib.pyplot as plt
import shared_functions as sf
from collections import Counter
import pandas as pd
from itertools import product

def fetch_characters(partisanship, hvv=None, hvv_tuple=None,whole_hvv=None, unordered=None):
    file = f'../sampled_pooled_alphabetized2/{partisanship}_data.json'

    data = sf.import_json(file)

    characters = []
    conversion = {'hero':0,'villain':1,'victim':2}
    for row in data:
        if hvv:
            characters.append(row[conversion[hvv]])
        elif hvv_tuple:
            characters.append("; ".join([row[conversion[hvv_tuple[0]]], row[conversion[hvv_tuple[1]]]]))
        elif whole_hvv:
            characters.append("; ".join(row[:3]))
        elif unordered:
            characters.append("; ".join(sorted(row[:3])))
        else:
            characters += row[:3]
    return [x for x in characters if x!='none' and x!='none; none' and x!='none; none; none']

def fetch_character_appearances(part, year, char):
    file = f'../sampled_pooled_alphabetized2/{part}_data.json'
    data = sf.import_json(file)
    char_data = [x for x in data if char in x]
    year_data = [x for x in char_data if f"{year}-" in x[5]]
    return len(year_data)


def convert_char_count_to_df(char_count, part):
    df = pd.DataFrame([char_count]).T.reset_index()
    df.columns = ['character',part]
    return df

def plot_common_characters_over_time(parts, top_chars):
    records = []
    for char in top_chars[:5]:
        for part in parts:
            record = {'character-partisanship':f"{char}-{part}"}
            for year in range(2016,2023):
                record[year]= fetch_character_appearances(part=part, year=year, char=char)
            records.append(record)
    df = pd.DataFrame(records)
    for i, row in df.iterrows():
        x=list(range(2016,2023))
        y=row[x].values.tolist()
        label=row['character-partisanship']
        plt.plot(x,y,label=label)
    plt.legend()
    plt.title(f"{', '.join(parts)}: Top Characters over Time")
    plt.show()




def plot_most_common_shared_characters(parts):

    dfs = []
    for part in parts:
        characters = fetch_characters(part)
        # num appearances for each, num of shared appearances
        dfs.append(convert_char_count_to_df(Counter(characters), part))

    df_a = dfs[0]
    merged = df_a.merge(dfs[1], on='character')
    if len(dfs) > 2:
        for df_b in dfs[2:]:
            merged = merged.merge(df_b, on='character')
    merged['sum'] = merged.iloc[:, 1:].sum(axis=1)
    sorted_ = merged.sort_values(by='sum', ascending=False)
    threshold = 20
    sorted_[:threshold].plot.bar()
    plt.xticks(ticks=list(range(len(sorted_[:threshold]['character'].values))),
               labels=list(sorted_[:threshold]['character'].values))
    plt.title(f"Most frequently mentioned characters among the {', '.join(parts)}")
    plt.tight_layout()
    plt.show()
    return list(sorted_[:threshold]['character'].values)

def plot_most_common_shared_archetypes(parts, h_v_v):
    dfs = []
    for part in parts:
        characters = fetch_characters(part, hvv=h_v_v)
        dfs.append(convert_char_count_to_df(Counter(characters), part))

    df_a = dfs[0]
    merged = df_a.merge(dfs[1], on='character')
    if len(dfs) > 2:
        for df_b in dfs[2:]:
            merged = merged.merge(df_b, on='character')
    merged['sum'] = merged.iloc[:, 1:].sum(axis=1)
    sorted_ = merged.sort_values(by='sum', ascending=False)
    threshold = 20
    sorted_[:threshold].plot.bar()
    plt.xticks(ticks=list(range(len(sorted_[:threshold]['character'].values))),
               labels=list(sorted_[:threshold]['character'].values))
    plt.title(f"Most frequently shared {h_v_v}s among the {', '.join(parts)}")
    plt.tight_layout()
    plt.show()
    return list(sorted_[:threshold]['character'].values)

def plot_most_common_partial_narratives(parts, a, b):
    dfs = []
    for part in parts:
        characters = fetch_characters(part, hvv_tuple=[a,b])
        dfs.append(convert_char_count_to_df(Counter(characters), part))

    df_a = dfs[0]
    merged = df_a.merge(dfs[1], on='character')
    if len(dfs) > 2:
        for df_b in dfs[2:]:
            merged = merged.merge(df_b, on='character')
    merged['sum'] = merged.iloc[:, 1:].sum(axis=1)
    sorted_ = merged.sort_values(by='sum', ascending=False)
    threshold = 20
    sorted_[:threshold].plot.bar()
    plt.xticks(ticks=list(range(len(sorted_[:threshold]['character'].values))),
               labels=list(sorted_[:threshold]['character'].values))
    plt.title(f"Most frequently shared {a}-{b} combos among the {', '.join(parts)}")
    plt.tight_layout()
    plt.show()
    return list(sorted_[:threshold]['character'].values)

def plot_most_common_complete_narratives(parts):
    dfs = []
    for part in parts:
        characters = fetch_characters(part,whole_hvv=True)
        dfs.append(convert_char_count_to_df(Counter(characters), part))

    df_a = dfs[0]
    merged = df_a.merge(dfs[1], on='character')
    if len(dfs) > 2:
        for df_b in dfs[2:]:
            merged = merged.merge(df_b, on='character')
    merged['sum'] = merged.iloc[:, 1:].sum(axis=1)
    sorted_ = merged.sort_values(by='sum', ascending=False)
    threshold = 20
    sorted_[:threshold].plot.bar()
    plt.xticks(ticks=list(range(len(sorted_[:threshold]['character'].values))),
               labels=list(sorted_[:threshold]['character'].values))
    plt.title(f"Most frequently shared narratives among the {', '.join(parts)}")
    plt.tight_layout()
    plt.show()
    return list(sorted_[:threshold]['character'].values)

def plot_most_common_unordered_narratives(parts):
    dfs = []
    for part in parts:
        characters = fetch_characters(part, unordered=True)
        dfs.append(convert_char_count_to_df(Counter(characters), part))

    df_a = dfs[0]
    merged = df_a.merge(dfs[1], on='character')
    if len(dfs) > 2:
        for df_b in dfs[2:]:
            merged = merged.merge(df_b, on='character')
    merged['sum'] = merged.iloc[:, 1:].sum(axis=1)
    sorted_ = merged.sort_values(by='sum', ascending=False)
    threshold = 20
    sorted_[:threshold].plot.bar()
    plt.xticks(ticks=list(range(len(sorted_[:threshold]['character'].values))),
               labels=list(sorted_[:threshold]['character'].values))
    plt.title(f"Most frequently shared unordered narratives among the {', '.join(parts)}")
    plt.tight_layout()
    plt.show()
    return list(sorted_[:threshold]['character'].values)
# Plot most common characters shared be each partisanship intersection
combos = [['FarRight','FarLeft'],
          sf.PARTISANSHIPS,
['CenterRight','Right','FarRight'],
['CenterLeft','Left','FarLeft'],
['CenterLeft','Center','CenterRight']
          ]
# for combo in combos:
#     top_char = plot_most_common_shared_characters(combo)
#     plot_common_characters_over_time(parts=combo, top_chars=top_char)

# for combo,hvv in product(combos, ['hero','villain','victim']):
#     plot_most_common_shared_archetypes(combo, hvv)

# for combo, hvv_a, hvv_b in product(combos, ['hero','villain','victim'], ['hero','villain','victim']):
#     if hvv_a==hvv_b:
#         continue
#     plot_most_common_partial_narratives(combo, hvv_a, hvv_b)

for combo in combos:
    # plot_most_common_complete_narratives(combo)
    plot_most_common_unordered_narratives(combo)