import shared_functions as sf
import pandas as pd
import  matplotlib.pyplot as plt

signal_data = sf.import_json('signals\\FarRight_signals_by_part_input_hvv.json')
#### FULL SIGNAL ###
# high_correlation_files = [x for x in sf.get_files_from_folder('cleaned_data_end_april', 'csv') if 'FarRight' in x]
#
# for file in high_correlation_files:
#     file_data = file.split('\\')[1].replace('.csv', '').split('_')
#     part_a = file_data[0]
#     input_level = file_data[1]
#     hvv = file_data[2]
#     part_b = file_data[3]
#
#     data = sf.import_csv(file)
#     for row in data:
#         character = row[0]
#         signal_pair = [x for x in signal_data[part_a][part_b][input_level][hvv] if character == x[0]][0]
#         plt.plot(list(range(168)),signal_pair[1],label=part_a)
#         plt.plot(list(range(168)),signal_pair[2],label=part_b)
#         plt.xlabel('Time (in 15 day periods)')
#         plt.ylabel('Number of articles')
#         plt.legend()
#         plt.title(f"{character} as {hvv}")
#         plt.show()
# print('')
### PARTIAL SIGNAL ##########
high_correlation_files = [x for x in sf.get_files_from_folder('cleaned_data_end_april_segments', 'csv') if 'FarRight' in x]

for file in high_correlation_files:
    file_data = file.split('\\')[1].replace('.csv', '').split('_')
    part_a = file_data[0]
    input_level = file_data[1]
    hvv = file_data[2]
    part_b = file_data[3]
    signal_length = int(file_data[4])

    data = sf.import_csv(file)
    for row in data:
        character = row[0]
        if character == 'none.none.none' or character=='joe biden':
            continue
        start = int(row[2])
        end = int(row[3])
        signal_pair = [x for x in signal_data[part_a][part_b][input_level][hvv] if character == x[0]][0]
        if max(signal_pair[1][start:end])<10 and max(signal_pair[2][start:end])<10:
            continue
        plt.plot(list(range(start, end)),signal_pair[1][start:end],label=part_a)
        plt.plot(list(range(start, end)),signal_pair[2][start:end],label=part_b)
        plt.xlabel('Time (in 15 day periods)')
        plt.ylabel('Number of articles')
        plt.legend()
        plt.title(f"{character} as {hvv}")
        plt.show()
print('')
