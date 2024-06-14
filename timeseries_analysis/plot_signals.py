import  shared_functions as sf
import matplotlib.pyplot as plt

def plot_signals(level, h_v_v, origin):
    output = sf.import_json(f'signals/{origin}_signals_by_part_input_hvv.json')[origin]


    for p_b in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
        if p_b == origin:
            continue
        rel_signals = output[p_b][level][h_v_v]
        for signal in rel_signals:
            plt.plot(signal[2])
        plt.title(f"{origin}, {p_b}")
        plt.show()

inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}

for input_level in inputs.keys():
    for hvv in inputs[input_level]:
        for source in ['FarRight','Right','CenterRight','Center','CenterLeft','Left','FarLeft']:
            plot_signals(input_level,hvv, source)
