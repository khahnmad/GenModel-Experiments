from sentence_transformers import SentenceTransformer
# import shared_functions as sf
import pickle
import json

def generate_template(heroes,villains, victims, template):
    if template=='a':
        prompt = f"The heroes are {', '.join(heroes)}" \
        f" and the villains are {', '.join(villains)} " \
        f"and the victims are {', '.join(victims)}."
    elif template=='b':
        prompt=f"Heroes: {', '.join(heroes)}; Villains: {', '.join(villains)}; Victims: {', '.join(victims)}"
    else:
        raise KeyError(f"'template' should be 'a' or 'b', not {template}")
    return prompt

def load_data(filename):
    with open(filename, 'r') as j:
        content = json.loads(j.read())
    return content



def apply_template(data, template_type):
    sentences = []
    for elt in data:
        template = generate_template(heroes=elt['denoising_result']['hero'],
                                     villains=elt['denoising_result']['villain'],
                                     victims=elt['denoising_result']['victim'],
                                     template=template_type)
        sentences.append(template)
    return sentences


def export_as_pkl(export_name:str, content):
    with open(export_name, "wb") as f:
        pickle.dump(content, f)
        f.close()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data = load_data('cluster_experiments/input/initial_subsample_results.json')
for temp in ['a','b']:
    sentences = apply_template(data, temp)
    sentence_embeddings = model.encode(sentences)
    count = 0
    output = []
    for sent, emb in zip(sentences,sentence_embeddings):
        row = {'sentence':sent,
               'embedding':emb,
               '_id':data[count]["_id"],
               'sample_id':data[count]['sample_id']}
        count +=1
        output.append(row)

    export_as_pkl(f'initial_subsample_{temp}.pkl', {"content":output,
                                                       'metadata': {"template_type":temp,
                                                                    "model": 'paraphrase-MiniLM-L6-v2'}})
