import shared_functions as sf
"""
{"FarRight": {"CenterLeft": {"single": {"hero"
"""
narrative_count = []
signal_files = sf.get_files_from_folder('signals','json')
for file in signal_files:
    data = sf.import_json(file)
    for p_a in data.keys():
        for p_b in data[p_a].keys():
            for input_level in data[p_a][p_b].keys():
                for hvv in data[p_a][p_b][input_level].keys():

