import shared_functions as sf

def find_between_part_intersection(a, b):
    # jaccard = len(set(a).intersection(set(b)))/(len(set(a))+len(set(b))-len(set(a).intersection(set(b))))
    overlap = len(set(a).intersection(set(b)))/min(len(set(a)),len(set(b)))
    return overlap

def fetch_fr_overlap():
    folder  = "C:\\Users\\khahn\\Documents\\Thesis\\timeseries data\\mainstreamed_text_narratives"
    intersections = []
    for file in sf.get_files_from_folder(folder,'csv'):
        key_data = file.split('narratives\\')[1].replace('.csv','').split('_')

        data = sf.import_csv(file)
        fr_heroes = [x[0] for x in data if x[3]=='FarRight']
        non_fr_heroes = [x[0] for x in data if x[3]!='FarRight']

        fr_villains = [x[1] for x in data if x[3]=='FarRight']
        non_fr_villains = [x[1] for x in data if x[3]!='FarRight']

        fr_vics = [x[2] for x in data if x[3]=='FarRight']
        non_fr_vics = [x[2] for x in data if x[3]!='FarRight']

        hero_intersection = find_between_part_intersection(a=fr_heroes,
                                                           b=non_fr_heroes)
        victim_intersection = find_between_part_intersection(a=fr_vics,
                                                           b=non_fr_vics)
        villain_intersection = find_between_part_intersection(a=fr_villains,b=non_fr_villains)
        intersections.append([key_data[0],key_data[2],key_data[-1],hero_intersection, villain_intersection, victim_intersection])

        print('')

print('')
"""
TODO: plot some variations of the overlap coeffiejnct to see if there's anything interesting there to talk about 
"""