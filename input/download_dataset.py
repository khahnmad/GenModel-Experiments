# import shared_functions as sf

import gridfs
from dotenv import load_dotenv
import os
import pymongo as pm

import pickle


def getConnection(
    connection_string: str = "", database_name: str = "", use_dotenv: bool = False
):
    "Returns MongoDB and GridFS connection"

    # Load config from config file
    if use_dotenv:
        load_dotenv()
        connection_string = os.getenv("CONNECTION_STRING")
        database_name = os.getenv("DATABASE_NAME")

    # Use connection string
    conn = pm.MongoClient(connection_string)
    db = conn[database_name]
    fs = gridfs.GridFS(db)

    return fs, db

def export_as_pkl(export_name:str, content):
    with open(export_name, "wb") as f:
        pickle.dump(content, f)
        f.close()

def query_collection(collection, query:dict, fields:list, n=None):
    fields = {k:1 for k in fields}


    docs = collection.find(query, fields)

    count, last_count = 0,0
    data = []
    for elt in docs:
        data.append(elt)
        if count % 1000 == 0:
            print(f'Exporting at {count}')
            export_as_pkl(f'complete_sample_{last_count}_{count}.pkl',
                              {'content':data,
                               'metadata': {'start':last_count,
                                            'end':count}})
            data = []
            last_count = count
        count += 1
    return docs

############################################# FULL SUBSAMPLE ####################################################
fs,db = getConnection(use_dotenv=True)

# Query for vectors
docs = query_collection(collection=db['sampled_articles'],
                           query={},
                           fields=['processing_result','denoising_result','embedding_result','sample_id','publish_date']
                           )

# Export sample
print(f'COMPLETE, exported {len(docs)}')