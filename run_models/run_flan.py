from explore_gen_models import humans_in_social_science_pipline as h
from run_models.support_scripts import fetch_data as f
import json
import time


def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def generate_prompt(hvv: str, data: dict, prompt_type: str, split_method: str):
    if prompt_type == 'A':
        prompt = "Output a response given the Output rules and Article.\nOutput Rules: Identify if" \
                 " there is one, multiple, or zero <elt>s in the article.\nIf the number of <elt>s == 0, then output " \
                 "'None'.\nIf the number of <elt>s > 0, then output the names of the <elt>s as a python list.\n" \
                 "Article: <article_text>"
    elif prompt_type == 'B':
        prompt = f"Who is the <elt> in the following text?\n<article_text>"
    template = prompt.replace('<elt>', hvv).split('<article')[0]
    w_data = prompt.replace('<article_text>', data['article_text']).replace('<elt>', hvv)
    return {'prompt': w_data, 'prompt_type': prompt_type, 'template': template, 'split_method': split_method}


def export_as_json(export_filename: str, output):
    with open(export_filename, "w") as outfile:
        outfile.write(json.dumps(output))


def prep_model(MODEL_NAME='google/flan-t5-base',):
    return h.TFG(model_name=MODEL_NAME, connect_to_gpu=True, memory_saver=True)

def run_model(data, model, MODEL_NAME='google/flan-t5-base', PROMPT_TYPE='B', SPLITTER_TYPE='langchain'):
    # model = h.TFG(model_name=MODEL_NAME, connect_to_gpu=True, memory_saver=True)
    annotations = []
    for i in range(len(data)):
        elt = data[i]
        duration = {k: None for k in ['hero', 'villain', 'victim']}
        for obj in ['hero', 'villain', 'victim']:
            prompt = generate_prompt(obj, elt, PROMPT_TYPE, SPLITTER_TYPE)

            a = time.time()
            response = model.generate(prompt)
            b = time.time()

            duration[obj] = b - a

            elt[f'model_{obj}'] = response[0]
            elt[f"model_{obj}_score"] = response[1]
            elt[f"prompt_length_exceeded"] = response[2]

        elt['duration'] = duration
        annotations.append(elt)

    return annotations



if __name__ == '__main__':
    MODEL_NAME = 'google/flan-t5-base'
    PROMPT_TYPE = 'B'
    SPLITTER_TYPE = 'langchain'

    sample_size = 50

    model = prep_model(MODEL_NAME)

    while True:
        a = time.time()
        data, start, end = f.fetch_data(sample_size)

        annotations = run_model(model=model, data=data)
        b = time.time()

        export_as_json(
            f"/home/kmadole/model_pipeline/true_output/{start}_{end}_annotations.json",
            {'content': data,
             'metadata': {'model': MODEL_NAME, 'prompt': PROMPT_TYPE, 'splitter': SPLITTER_TYPE,
                          'start_date':start, 'end_date':end}})

        print(f"Exporting {len(annotations)}, took {(b-a)/60} minutes")