import shared_functions as sf

# full_signal = sf.get_files_from_folder('cleaned_data_end_april_monthbins','csv')
# exportable = []
# for file in full_signal:
#     file_data = file.split('end_april_monthbins\\')[1].replace('.csv','').split('_')
#     data = sf.import_csv(file)
#
#     exportable += [{'first_partisanship': file_data[0],
#       'lagging_partisanship': file_data[-1],
#       'hvv_combination_type': file_data[1],
#       'hvv_character_type': file_data[2],
#       'narrative': data[i][0]} for i in range(len(data))]
# sf.export_as_json('narrative_matches_months.json',exportable)
pooled_files = sf.get_files_from_folder('sampled_pooled_alphabetized2','json')
narrative_matches = sf.import_json('narrative_matches_months.json')
narrative_matches = [x for x in narrative_matches if x['hvv_combination_type']=='combo']
results = []
for p in sf.PARTISANSHIPS:
    rel_rows = [x for x in narrative_matches if x['first_partisanship']==p and x['narrative']!='none.none.none']
    file = [x for x in pooled_files if f"\\{p}_" in x][0]
    pooled_data = sf.import_json(file)

    for row in rel_rows:
        results += [x for x in pooled_data if f"{x[0]}.{x[1]}.{x[2]}".lower()==row['narrative']]

for p in sf.PARTISANSHIPS:
    rel_rows = [x for x in narrative_matches if x['lagging_partisanship'] == p and x['narrative']!='none.none.none']
    file = [x for x in pooled_files if f"\\{p}_" in x][0]
    pooled_data = sf.import_json(file)

    for row in rel_rows:
        results += [x for x in pooled_data if f"{x[0]}.{x[1]}.{x[2]}".lower() == row['narrative']]
sf.export_as_json('matched_article_ids.json', {'content':results})
print('')