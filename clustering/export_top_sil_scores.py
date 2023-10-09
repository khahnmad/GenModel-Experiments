import shared_functions as sf
import pandas as pd

def fetch_relevant_files(hvv_temp:str, cluster_type:str,  vers=0, single_combo='single'):
    if vers == 0:
        opt_loc = f'./clustering_optimizations/version_0'
        if single_combo == 'single':
            opt_loc += f'/separate'
        else:
            opt_loc += f'/combined'

        optimizations = sf.get_files_from_folder(opt_loc, 'json')

    else:
        opt_loc = f'./clustering_optimizations/version_{vers}'
        if single_combo == 'single':
            opt_loc += f'/separate'
        else:
            opt_loc += f'/combined'

        optimizations = sf.get_files_from_folder(opt_loc, 'json')

    relevant = [x for x in optimizations if cluster_type in x and hvv_temp in x]
    if len(relevant) == 0:
        print(f"No files found for {hvv_temp}, {cluster_type}, v{vers}")
        return None
    return relevant

def import_optimization(hvv_temp:str, cluster_type:str, vers=0, single_combo='single'):
    # Fetch the relevant optimization files
    relevant_files = fetch_relevant_files(hvv_temp, cluster_type, vers, single_combo)
    if relevant_files is None:
        return

    # Import the data from the relevant files
    data = []
    for file in relevant_files:
        content = sf.import_json(file)
        data += content['content'][1:]
        columns = content['content'][0]

    # Format the data into a df
    df = pd.DataFrame(data=data, columns=columns)
    return df['silhouette_score'].max()

############################ ACTION #########################################
combo_table = [['cluster type', 'template','version','high score']]
for cluster_type in ['dbscan','kmeans','agglom']:
    for template in ['combo_a','combo_b','combo_c','combo_d']:
        for version in range(3):
            high_score = import_optimization(hvv_temp=template, cluster_type=cluster_type,vers=version,
                                             visualize=True, single_combo='combo')
            combo_table.append([cluster_type, template[-1], version, high_score])
sf.export_nested_list('Combo_Max_Sil_Scores.csv',combo_table)

sep_table = [['cluster type', 'hvv','version','high score']]
for cluster_type in ['dbscan','kmeans','agglom']:
    for hvv in ['hero','villain','victim']:
        for version in range(3):
            high_score = import_optimization(hvv_temp=hvv, cluster_type=cluster_type,vers=version,
                                             single_combo='single')
            sep_table.append([cluster_type, hvv, version, high_score])
sf.export_nested_list('Sep_Max_Sil_Scores.csv',sep_table)