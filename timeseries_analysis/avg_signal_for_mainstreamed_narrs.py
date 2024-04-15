"""
TODO: Run with complete FR final quantity narratives
"""
import shared_functions as sf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

def is_mainstreamed(character, partisanship):
    mainstreamed_narrs = sf.import_csv('tables/Final_quantity_narratives.csv')
    for row in mainstreamed_narrs:
        if character==row[-1] and partisanship==row[3]:
            return True
    return False

other_data = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\FarRight_signals_by_part_input_hvv.json')
data = sf.import_json('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\signals\\signals_by_part_input_hvv.json')
print('')

source = data['FarRight']

plottable = [['InputType','part','0>=x>0.25','0.25>=x>0.5','0.5>=x>0.75','0.75>=']]
fr_signals, c_signals = [],[]
for p in source.keys():
    if "Center" in p: # Ignoring the different types of Centrist
        for input_type in source[p].keys():
            for hvv in source[p][input_type]:
                for c_signal_pair in source[p][input_type][hvv]:
                    if is_mainstreamed(c_signal_pair[0],p):
                        fr_signals.append(c_signal_pair[1])
                        c_signals.append(c_signal_pair[2])

fr_avg = np.average(fr_signals, axis=0)
c_avg = np.average(c_signals, axis=0)
x = range(len(fr_avg))
plt.plot(x, fr_avg,label='Far Right')
plt.plot(x, c_avg,label='Centrist')
plt.xlabel('Time')
plt.ylabel('# of Narrative Occurrences')
plt.legend()
plt.show()