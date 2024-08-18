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
def generate_matrix(year):
    hvv = 'hero, villain, victim'
    matrix = []
    _min , _max =100, -100
    for part_a in sf.PARTISANSHIPS:
        for part_b in sf.PARTISANSHIPS:
            if part_a == part_b:
                continue
            if year:
                file  = f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_monthbins\\{year}\\{part_a}_combo_{hvv}_{part_b}.csv"
            else:
                file =  f"C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_monthbins\\{part_a}_combo_{hvv}_{part_b}.csv"
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

    grouped_leading = df.loc[df.groupby('A')['Weight'].idxmax()].to_dict('records')
    grouped_lagging = df.loc[df.groupby('B')['Weight'].idxmax()].to_dict('records')
    final =pd.DataFrame(grouped_leading+grouped_lagging)
    final['A'] = pd.Categorical(final['A'], ["FarLeft", "Left", "CenterLeft", 'Center', 'CenterRight', 'Right', 'FarRight'])
    final = final.drop_duplicates()
    # final = final[final['Weight']>=0.6]
    final = final.sort_values(by='A')
    # final['Weight'] = final['Weight']
    # df = df[df['Weight']>=7]


    G = nx.from_pandas_edgelist(final, source='A',target='B',edge_attr='Weight', create_using=nx.DiGraph())
    # pos = nx.lay
    color_map = []
    color_conversion = {'FarRight':'#ff0000',
                        'Right':'#ff7400',
                        'CenterRight':'#ffb100',
                        'Center':'#ffe600',
                        'CenterLeft':'#00c582',
                        'Left':'#008292',
                        'FarLeft':'#1400ff'}
    level_conversion = {'FarRight':3,
                        'Right':2,
                        'CenterRight':1,
                        'Center':0,
                        'CenterLeft':1,
                        'Left':2,
                        'FarLeft':3}
    for node in G.nodes:
        color_map.append(color_conversion[node])

    nx.set_node_attributes(G, level_conversion, 'level')

    nx.draw(G,
            with_labels=True,
            arrows=True,
            width=df.Weight.values,
            node_color=color_map,
            # pos=nx.multipartite_layout(G,subset_key='level')
            pos=nx.circular_layout(G)
            )

    plt.show()
    # nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
    # plt.show()
    # save graph object to file
    # pickle.dump(G, open(f'part_influence_as_network.pickle', 'wb'))
    # write_and_export(f'part_influence_as_network', G)





if __name__ == '__main__':
    # for yr in range(2016,2023):
    #     matrix = generate_matrix(yr)
    #     generate_network(matrix)
    matrix = generate_matrix(None)
    generate_network(matrix)
#todo: add colors according to part
# todo: make them stay in the same place each yaear
"""
the problem is that there is not slow movement from yaer to year - it feels totally random each time 
"""