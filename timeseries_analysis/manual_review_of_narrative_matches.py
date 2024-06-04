import shared_functions as sf
import bson
import matplotlib.pyplot as plt

SIGNAL = sf.import_json('signals/FarRight_signals_by_part_input_hvv.json')

def extract_key_data(filename):
    if 'segments' in filename:
        key_data = filename.split('end_april_segments\\')[1].replace('.csv', '').split('_')
        return key_data[0], key_data[-2], key_data[1], key_data[2]
    else:
        key_data = filename.split('end_april\\')[1].replace('.csv','').split('_')
        return key_data[0], key_data[-1], key_data[1], key_data[2]

def sample_text(narr, part_a, part_b, input_level, hvv, segment_range):
    fs,db = sf.getConnection()

    part_a_file = [x for x in sf.get_files_from_folder('sampled_pooled_alphabetized2','json') if f"\\{part_a}_" in x][0]
    part_b_file = [x for x in sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json') if f"\\{part_b}_" in x][0]

    hvv_indexing = {'hero':0,'villain':1,'victim':2}

    data = sf.import_json(part_a_file) + sf.import_json(part_b_file)

    if input_level=='combo':
        narr_data = narr.split('.')
        rel_data = [x for x in data if x[0]==narr_data[0] and x[1]==narr_data[1] and x[2]==narr_data[2]]
    elif input_level=='tuple':
        print('NOT YET SUPPORTED')

    else:
        rel_data = [x for x in data if x[hvv_indexing[hvv]] == narr]

    rel_ids = [bson.ObjectId(x[3]) for x in rel_data]
    docs = db['sampled_articles'].find({'_id':{"$in":rel_ids}})
    for doc in docs:
        text = doc['parsing_result']
        print('')
    return

def articles_meet_threshold(part_a, part_b, input_level, hvv, narrative, threshold):
    try:
        match = [x for x in SIGNAL[part_a][part_b][input_level][hvv] if x[0] == narrative][0]
    except IndexError:
        return \
            False
    except KeyError:
        print(f"{part_a}, {part_b}, {input_level}, {hvv}")
    # if sum(match[1])>=threshold and sum(match[2])>=threshold and  sum(match[1])<15 and sum(match[2])<15:
    if sum(match[1]) >= threshold and sum(match[2]) >= threshold:
        return True
    print(f"{narrative}: Only {sum(match[1])}, {sum(match[2])}")
    return False

def plot_signal(part_a, part_b, input_level, hvv, narrative, seg_length=None):
    match = [x for x in SIGNAL[part_a][part_b][input_level][hvv] if x[0] == narrative][0]
    if seg_length:
        plt.plot(list(seg_length), match[1][seg_length[1]:seg_length[-1]+2], label=part_a)
        plt.plot(list(seg_length), match[2][seg_length[1]:seg_length[-1]+2], label=part_b)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mentions')
        plt.title(f'{narrative} as {hvv}: {part_a}->{part_b}')
        plt.show()
    else:
        plt.plot(range(len(match[1])),match[1],label=part_a)
        plt.plot(range(len(match[1])),match[2],label=part_b)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mentions')
        plt.title(f'{narrative} as {hvv}: {part_a}->{part_b}')
        plt.show()

complete_files = [x for x in sf.get_files_from_folder(
    'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april', 'csv')]
partial_files = [x for x in sf.get_files_from_folder(
    'C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\timeseries_analysis\\cleaned_data_end_april_segments',
    'csv')]
narrative = 'ukraine'
part_a = 'FarRight'
part_b = 'CenterRight'
input_level = 'single'
hvv = 'victim'
segment_length = range(0,120)
# sampled_text = sample_text(narrative, part_a, part_b, input_level, hvv, segment_length)

# Let's look first at high quantity narratives + combo narratives
threshold = 10
already = []
for file in partial_files+complete_files:
    part_a, part_b, input_level, hvv = extract_key_data(file)
    if input_level!='combo':
        continue
    data = sf.import_csv(file)
    for row in data:
        narrative = row[0]
        if narrative in already:
            continue
        if narrative=='none.none.none':
            continue
        if 'segments' in file:
            segment_range = range(int(row[2]),int(row[3])+1)
        else:
            segment_range = range(0,175)
        # find the number of articles talking about this
        if articles_meet_threshold(part_a, part_b, input_level, hvv, narrative, threshold) is False:
            continue
        plot_signal(part_a, part_b, input_level, hvv, narrative, segment_range)
        # sampled_text = sample_text(narrative, part_a, part_b,input_level, hvv)
        already.append(narrative)
        print('')
        # how many articles are there? take a sample and then look to see if they're talking about the same thing?

