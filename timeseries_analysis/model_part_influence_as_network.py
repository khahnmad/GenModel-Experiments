"""
Matrix:
part_a, part_b, a->b influence

Network:
Show only the strongest values? It will be a fully connected network
"""
import shared_functions as sf
import networkx as nx
import pandas as pd
from pyvis.network import Network
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pickle

def min_max_normalize(value, _min, _max):
    return (value-_min)/(_max-_min)

# import influence values for each partisanship
# should be the same data as used for the heatmaps
def generate_matrix():
    hvv = 'hero, villain, victim'
    matrix = []
    _min , _max =100, -100
    for part_a in sf.PARTISANSHIPS:
        for part_b in sf.PARTISANSHIPS:
            if part_a == part_b:
                continue
            file  = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_monthbins\\{part_a}_combo_{hvv}_{part_b}.csv"
            data = sf.import_csv(file)
            # deciding that the weight between partisnaships will be the number of narratives that they included - this is the same as what i used for the heatmpap
            matrix.append([part_a, part_b, len(data)])
            if len(data)<_min:
                _min = len(data)
            if len(data)>_max:
                _max = len(data)
    # for row in matrix:
    #     row[-1] = min_max_normalize(row[-1], _min, _max)
    #     print('')
    return matrix



def write_and_export(title, graph):
    net = Network(notebook=True,cdn_resources='in_line')
    net.repulsion()
    net.from_nx(graph)
    html = net.generate_html()
    with open(f"{title}.html", mode='w', encoding='utf-8') as fp:
        fp.write(html)
    display(HTML(html))


def generate_network(data):
    df = pd.DataFrame(data=data, columns=['A', 'B', 'Weight'])
    df = df[df['Weight']>=7]

    G = nx.from_pandas_edgelist(df, source='A',target='B',edge_attr='Weight', create_using=nx.DiGraph())
    nx.draw(G, with_labels=True,arrows=True)
            # , width=df.Weight.values)
    plt.show()
    # nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
    # plt.show()
    # save graph object to file
    # pickle.dump(G, open(f'part_influence_as_network.pickle', 'wb'))
    # write_and_export(f'part_influence_as_network', G)





if __name__ == '__main__':
    matrix = generate_matrix()
    generate_network(matrix)
