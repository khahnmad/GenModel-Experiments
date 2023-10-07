import json
import glob
import csv
import gridfs
from dotenv import load_dotenv
import os
import pymongo as pm
from bson import json_util
import pickle


def import_pkl_file(file):
    with open(file, "rb") as f:
        pkl_file = pickle.load(f)
        f.close()
    return pkl_file


def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def export_as_json(export_filename:str, output):
    if export_filename.endswith('.json') is not True:
        raise Exception(f"{export_filename} should be a .json file")
    try:
        with open(export_filename, "w") as outfile:
            outfile.write( json.dumps(output))
    except TypeError:
        with open(export_filename, "w") as outfile:
            outfile.write( json_util.dumps(output))


def import_csv(csv_file:str):
    # Given a file location, imports the data as a nested list, each row is a new list

    nested_list = []  # initialize list
    with open(csv_file, newline='', encoding='utf-8') as csvfile:  # open csv file
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            nested_list.append(row)  # add each row of the csv file to a list

    return nested_list

def get_files_from_folder(folder_name:str, file_endings:str)->list:
    return [x for x in glob.glob(folder_name + f"/*.{file_endings}")]

def remove_duplicates(data:list)->list:
    new_list  =[]
    for elt in data:
        if elt not in new_list:
            new_list.append(elt)
    return new_list


def export_nested_list(csv_name:str, nested_list):
    # Export a nested list as a csv with a row for each sublist
    with open(csv_name, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in nested_list:
            writer.writerow(row)

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

def query_collection(collection, query:dict, fields:list, n=None):
    fields = {k:1 for k in fields}
    if n:
        docs = list(collection.find(query, fields).limit(100))
    else:
        docs = list(collection.find(query, fields))
    return docs


def export_as_pkl(export_name:str, content):
    with open(export_name, "wb") as f:
        pickle.dump(content, f)
        f.close()