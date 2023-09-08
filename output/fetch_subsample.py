import shared_functions as sf
import datetime
fs, db = sf.getConnection(use_dotenv=True)


def fetch_subsample():
    cursor = list(db['sampled_articles'].find({'initial_subsample': True}))

    missing = 0
    for elt in cursor:
        if 'processing_result' not in elt.keys():
            missing+=1
    print(f"{missing} documents do not have a processing result yet")
    sf.export_as_json('subsample_processing_results.json',{'content':cursor,
                                                           'metadata': {"last_run": datetime.datetime.timestamp(datetime.datetime.today())}})

def fetch_sample():
    cursor = db['sampled_articles'].find()
    print('Query Complete')
    cursor = list(cursor)
    print('Listify compelete')
    # missing = 0
    # for elt in cursor:
    #     if 'processing_result' not in elt.keys():
    #         missing += 1
    # print(f"{missing} documents do not have a processing result yet")
    sf.export_as_json('sample_processing_results.json', {'content': cursor,
                                                            'metadata': {"last_run": datetime.datetime.timestamp(
                                                                datetime.datetime.today())}})
    print('Export Complete')


fetch_sample()
# fetch_subsample()