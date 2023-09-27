import shared_functions as sf
import time
from string import punctuation
from nltk.corpus import stopwords
import nltk

STOPWORDS = stopwords.words('english')
def import_data(sub):
    path = "C:\\Users\\khahn\\Documents\\Github\\GenModel-Experiments\\output\\"
    if sub:
        data = sf.import_json(path+'subsample_processing_results.json')['content']
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


def prep_cleaned_matches(sub:bool):
    data = import_data(sub)

    prev_file =  sf.import_json('cleaned_exact_matches_5000.json')
    outcome = prev_file['content']
    # flattened_outcome = flatten_existing_dict(outcome)
    start = prev_file['metadata']['limit']+1
    for i in range(start, len(data)):
        if 'processing_result' not in data[i].keys():
            continue
        result = data[i]['processing_result']
        hs = [clean(h) for h in sf.remove_duplicates(result['hero'])]
        vils = [clean(vil) for vil in sf.remove_duplicates(result['villain'])]
        vics = [clean(vic) for vic in sf.remove_duplicates(result['victim'])]

        for h in hs:
            for vil in vils:
                for vic in vics:
                    outcome.append([h,vil, vic,data[i]["_id"], data[i]['partisanship'], data[i]['publish_date']])

        if str(i).endswith('500') or str(i).endswith('000'):

            sf.export_as_json(f'cleaned_exact_matches_{i}.json', {'content': outcome,
                                                          'metadata': {'time': time.time(),
                                                                       'limit': i}})
            print(f'Exported checkpoint {i}')
    # pruned = {}
    # for k in outcome.keys():

    #     for j in outcome[k].keys():
    #         for l in outcome[k][j].keys():
    #             if len(outcome[k][j][l]) > 1:
    #                 pruned[k] = outcome[k]
    sf.export_as_json('cleaned_exact_matches.json', {'content': outcome,
                                             'metadata': time.time()})

# prep_exact_matches(sub=True)
prep_cleaned_matches(sub=True)