import json

def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def export_as_json(export_filename:str, output):
    if export_filename.endswith('.json') is not True:
        raise Exception(f"{export_filename} should be a .json file")
    try:
        with open(export_filename, "w") as outfile:
            outfile.write(output)
    except TypeError:
        response = input("'output' should be a str, not dict. Do you want to convert this using json.dumps()? Y/N")
        if response.lower()=='y':
            export_as_json(export_filename, json.dumps(output))
