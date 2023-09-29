import shared_functions as sf
import time
from string import punctuation
from nltk.corpus import stopwords
import nltk

STOPWORDS = stopwords.words('english')
def import_data(sub, version=0):
    path = "C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\output\\"
    if sub:
        if version==0:
            data = sf.import_json(path+'subsample_processing_results.json')['content']
        elif version ==1:
            data = sf.import_json("C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\input\\initial_subsample_triplets_results.json")
        else:
            raise Exception(f"{version}: incorrect version number")
    else:
        data = sf.import_json(path+'sample_processing_results.json')
    return data


def get_hvv_combos(data:dict, w_cleaning:bool)->list:
    combos = []
    for h in data['hero']:
        for vil in data['villain']:
            for vic in data['victim']:
                combos.append([h, vil, vic])
    return combos


def add_metadata(hvv_data:list, _id:str, partisanship:str, date:str)->list:
    for elt in hvv_data:
        elt += [_id, partisanship, date]
    return hvv_data

def clean(value:str)->str:
    lower = value.lower()
    for elt in punctuation:
        lower = lower.replace(elt,'')
    tokens = nltk.word_tokenize(lower)
    no_stops = [x for x in tokens if x not in STOPWORDS]
    return " ".join(no_stops)



def prep_exact_matches(sub:bool):
    data = import_data(sub)

    prev_file = sf.import_json('exact_matches_7500.json')
    outcome = prev_file['content']
    start = prev_file['metadata']['limit']+1
    for i in range(start, len(data)):
        if 'processing_result' not in data[i].keys():
            continue
        result = data[i]['processing_result']

        for h in sf.remove_duplicates(result['hero']):
            if h not in outcome.keys():
                outcome[h] = {}
            for vil in sf.remove_duplicates(result['villain']):
                if vil not in outcome[h].keys():
                    outcome[h][vil]={}
                for vic in sf.remove_duplicates(result['victim']):
                    if vic not in outcome[h][vil].keys():
                        outcome[h][vil][vic] = []
                    outcome[h][vil][vic].append([data[i]["_id"],data[i]['partisanship'], data[i]['publish_date']])
        if str(i).endswith('500'):
            pruned = {}
            for k in outcome.keys():
                for j in outcome[k].keys():
                    for l in outcome[k][j].keys():
                        if len(outcome[k][j][l])>1:

                            if k not in pruned.keys():
                                pruned[k] = {}
                            if j not in pruned[k].keys():
                                pruned[k][j] = {}
                            if l not in pruned[k][j].keys():
                                pruned[k][j][l] = []
                            pruned[k][j][l] = outcome[k][j][l]
            sf.export_as_json(f'exact_matches_{i}.json', {'content': pruned,
                                                     'metadata': {'time': time.time(),
                                                                  'limit':i}})
    pruned = {}
    for k in outcome.keys():
        for j in outcome[k].keys():
            for l in outcome[k][j].keys():
                if len(outcome[k][j][l]) > 1:
                    pruned[k] = outcome[k]
    sf.export_as_json('exact_matches.json', {'content': pruned,
                                             'metadata': time.time()})
    # remove those

def flatten_existing_dict(data):
    flattened = []
    for h in data.keys():
        for vil in data[h].keys():
            for vic in data[h][vil]:
                # [data[i]["_id"], data[i]['partisanship'], data[i]['publish_date']]
                for elt in data[h][vil][vic]:
                    flattened.append([h,vil,vic]+elt)
    return flattened

def alphabetize_existing_dict(data):
    alphabetized = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [], 'i': [], 'j': [], 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [], 'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}
    for h in data.keys():
        if len(h)>0:
            letter = h[0].lower()
        else:
            letter = ''
        if letter not in alphabetized.keys():
            alphabetized[letter] = []

        for vil in data[h].keys():
            for vic in data[h][vil]:
                for elt in data[h][vil][vic]:
                    alphabetized[letter].append([h,vil,vic]+elt)
    return alphabetized

def fetch_alphabetized_content():
    alphabet = {}
    files = [x for x in sf.get_files_from_folder('.','json') if '_cleaned_exact_matches.json' in x]
    for file in files:
        if len(file)==30:
            letter = file[2]
        else:
            letter = ''
        alphabet[letter] =[]
    # import a
    start = sf.import_json('a_cleaned_exact_matches.json')['metadata']['limit']+1
    print(f'Starting at {start}')
    return alphabet, start

def prep_cleaned_matches(sub:bool, version=0):
    data = import_data(sub,version)

    # prev_file =  sf.import_json('cleaned_exact_matches_4500.json')
    # outcome = prev_file['content']
    # flattened_outcome = alphabetize_existing_dict(outcome)
    # flattened_outcome, start  = fetch_alphabetized_content()
    # start = prev_file['metadata']['limit']+1
    start = 0
    flattened_outcome = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [], 'i': [], 'j': [], 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [], 'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}
    for i in range(start, len(data)):
        # print(i)
        if 'processing_result' not in data[i].keys():
            continue
        if 'denoising_result' in data[i].keys():
            result = data[i]['denoising_result']
        else:
            result = data[i]['processing_result']
        hs = [clean(h) for h in sf.remove_duplicates(result['hero'])]
        vils = [clean(vil) for vil in sf.remove_duplicates(result['villain'])]
        vics = [clean(vic) for vic in sf.remove_duplicates(result['victim'])]

        for h in hs:
            if len(h) > 0:
                letter = h[0].lower()
            else:
                letter = ''
            if letter not in flattened_outcome.keys():
                flattened_outcome[letter] = []

            for vil in vils:
                for vic in vics:
                    flattened_outcome[letter].append([h,vil, vic,data[i]["_id"], data[i]['partisanship'], data[i]['publish_date']])

        if str(i).endswith('000'):
            print('Starting export')
            start_time = time.time()
            for l in flattened_outcome.keys():
                filename = f'alphabet_v1/{l}_cleaned_exact_matches.json'
                try:
                    existing = sf.import_json(filename)['content']
                    sf.export_as_json(filename, {'content': flattened_outcome[l]+existing,
                                                 'metadata': {'time': time.time(),
                                                              'limit': i}})
                except FileNotFoundError:
                    sf.export_as_json(filename, {'content': flattened_outcome[l],
                                                          'metadata': {'time': time.time(),
                                                                       'limit': i}})
                # print(f'exported {l}')
            end_time = time.time()
            print(f'Exported checkpoint {i}, took {(end_time-start_time)/60} minutes')
            flattened_outcome = {k:[] for k in flattened_outcome.keys()} # Clean the dictionary to start over
    # pruned = {}
    # for k in outcome.keys():

    #     for j in outcome[k].keys():
    #         for l in outcome[k][j].keys():
    #             if len(outcome[k][j][l]) > 1:
    #                 pruned[k] = outcome[k]
    for l in flattened_outcome.keys():
        filename = f'alphabet_v1/{l}_cleaned_exact_matches.json'
        try:
            existing = sf.import_json(filename)['content']
            sf.export_as_json(filename, {'content': flattened_outcome[l] + existing,
                                         'metadata': {'time': time.time(),
                                                      'limit': 'complete'}})
        except FileNotFoundError:
            sf.export_as_json(filename, {'content': flattened_outcome[l],
                                         'metadata': {'time': time.time(),
                                                      'limit': 'complete'}})


# prep_exact_matches(sub=True)
prep_cleaned_matches(sub=True, version=1)