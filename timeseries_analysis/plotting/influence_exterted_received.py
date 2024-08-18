import pandas as pd
import matplotlib.pyplot as plt
import shared_functions as sf

loc = r'C:\Users\khahn\Documents\Github\GenModel-Experiments\timeseries_analysis\cleaned_data_end_april_monthbins'
files = [x for x in sf.get_files_from_folder(loc, 'csv') if 'combo' in x]
data = {'Influencer':[],
        'Influenced':[],
        '# of Narratives':[]}

for file in files:
    part_a = file.split('bins\\')[1].split('_combo')[0]
    part_b = file.split('victim_')[1].replace('.csv','')
    csv_data = sf.import_csv(file)
    data['Influencer'].append(part_a)
    data["Influenced"].append(part_b)
    data['# of Narratives'].append(len(csv_data))

df = pd.DataFrame(data)


# Calculate total influence exerted by each person
influence_exerted = df.groupby('Influencer')['# of Narratives'].sum()

# Calculate total influence received by each person
influence_received = df.groupby('Influenced')['# of Narratives'].sum()

# Plot bar charts
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Influence exerted
ax[0].bar(influence_exerted.index, influence_exerted.values, color='blue')
ax[0].set_title('Total Influence Exerted')
ax[0].set_xlabel('Influencer')
ax[0].set_ylabel('# of Narratives')

# Influence received
ax[1].bar(influence_received.index, influence_received.values, color='green')
ax[1].set_title('Total Influence Received')
ax[1].set_xlabel('Influenced')
ax[1].set_ylabel('# of Narratives')

plt.show()
