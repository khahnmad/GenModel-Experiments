import json
import matplotlib.pyplot as plt
import pandas as pd


def import_json(file: str) -> dict:
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content

def plot_data(hvv, name):
    for p in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
        x, x_ticks, y = [], [], []
        count = 0
        for yr in hvv_dict[hvv][name][p].keys():
            x.append(int(yr))
            num_articles = sum(hvv_dict[hvv][name][p][yr].values())
            total_articles = sum(time_part_dict[p][yr].values())
            y.append(num_articles / total_articles)
        plt.plot(x, y, label=p)

    plt.legend()
    plt.title(
        f"{name} as a {hvv}: Percentage of Articles with an Appearance of the keyword over month by partisanship")
    try:
        plt.savefig(f'output/{name}_{hvv}.png', bbox_inches='tight')
    except OSError:
        print(name, hvv)
    plt.show()
data = import_json('person_role_data.json')
hvv_dict = data['person_role_count']
time_part_dict = data['total_count']

print('')
for role in hvv_dict.keys():
    for person in hvv_dict[role].keys():
        plot_data(role, person)