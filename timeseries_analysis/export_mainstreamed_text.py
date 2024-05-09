import shared_functions as sf

full_signal = sf.get_files_from_folder('cleaned_data_end_april','csv')
exportable = []
for file in full_signal:
    file_data = file.split('end_april\\')[1].replace('.csv','').split('_')
    data = sf.import_csv(file)

    exportable += [{'first_partisanship': file_data[0],
      'lagging_partisanship': file_data[-1],
      'hvv_combination_type': file_data[1],
      'hvv_character_type': file_data[2],
      'narrative': data[i][0]} for i in range(len(data))]
sf.export_as_json('narrative_matches.json',exportable)
