import shared_functions as sf
import pandas as pd

files = sf.get_files_from_folder('C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\15day_top_narratives','json')

data = []
for file in files:
    data += sf.import_json(file)

df = pd.DataFrame(data)
no_nones = df[df['combo']!='none.none.none']
sorted_no_nones = no_nones.sort_values(by='normalized_appearances', ascending=False)
value_counts = no_nones['combo'].reset_index().value_counts()
print('')