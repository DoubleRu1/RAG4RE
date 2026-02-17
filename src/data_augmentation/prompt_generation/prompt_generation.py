import os
import sys
import json

import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from prompt_templates import get_zero_shot_template_tacred, get_zero_shot_template_tacred_rag, semeval_prompt_template_rag, semeval_prompt_template

def read_json(path):
    """Read json file"""

    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    return data

def write_json(path, data):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print(path)
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def _text_to_tokens(line):
    if isinstance(line.get("tokens"), list):
        return [str(t) for t in line["tokens"]]
    if isinstance(line.get("tokens"), str):
        return line["tokens"].split()

    text = (
        line.get("sentence")
        or line.get("text")
        or line.get("sent")
        or line.get("context")
        or ""
    )
    if not text:
        return []
    return str(text).split()

def _extract_entity(line, primary_keys, fallback_entities, fallback_idx):
    for key in primary_keys:
        value = line.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), line.get(f"{key}_type", "")
        if isinstance(value, dict):
            name = value.get("text") or value.get("name") or value.get("mention")
            etype = value.get("type", "")
            if name:
                return str(name), str(etype)

    entities = line.get(fallback_entities, [])
    if isinstance(entities, list) and len(entities) > fallback_idx and isinstance(entities[fallback_idx], dict):
        ent = entities[fallback_idx]
        name = ent.get("text") or ent.get("name") or ent.get("mention")
        etype = ent.get("type", "")
        if name:
            return str(name), str(etype)

    text = " ".join(_text_to_tokens(line))
    if "<e1>" in text and "<e2>" in text:
        if fallback_idx == 0:
            m = re.findall("<e1>(.*?)</e1>", text, re.DOTALL)
        else:
            m = re.findall("<e2>(.*?)</e2>", text, re.DOTALL)
        if m:
            return " ".join(m).strip(), ""

    return ("HEAD" if fallback_idx == 0 else "TAIL"), ""

def normalize_relation_example(line):
    tokens = _text_to_tokens(line)
    subject, subject_type = _extract_entity(
        line,
        ["subject", "subj", "head", "entity1", "arg1"],
        "entities",
        0,
    )
    obj, object_type = _extract_entity(
        line,
        ["object", "obj", "tail", "entity2", "arg2"],
        "entities",
        1,
    )
    relation = line.get("relation") or line.get("label") or line.get("rel") or "no_relation"

    return {
        "tokens": tokens,
        "subject": subject,
        "object": obj,
        "relation": relation,
        "subject_type": subject_type,
        "object_type": object_type,
    }

def tacred_format(test_data, relations, similar_sentences, type="rag"):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []
    relations = " ".join([relation for relation in relations])

    
    for index, line in enumerate(test_data):
        
        sentence = " ".join([ token for token in line['tokens']])
        head = line['subject']
        tail = line['object']

        if  type == "simple":
            prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
            data = {"prompt":prompt, "relation":line['relation']}
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'])
            data = {"prompt":prompt, "relation":line['relation']}
        prompts.append(data)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts

def semeval_format(test_data, relations, similar_sentences, prompt_type="simple"):
    """Regenerate prompt for semeval dataset

    Args:
        test_data (list): list of test sentences along with e1 and e2
        relations (list): target relation label indexes
        similar_sentences (list): list of similar sentence with corresponding test data
        prompt_type (str, optional): prompt type. Defaults to "simple".

    Returns:
        list: the list of regenerated prompts
    """
    
    relation_names = list(set(relations))
    labels = relations
    relations = ", ".join([relation for relation in relation_names])
    prompts = []

    for index, line in enumerate(test_data):

        label = labels[index]
        sentence = line
        if len(similar_sentences) == 0:
            context = ''
        else:
            context = similar_sentences[index]

        e1_index = sentence.find("<e1>")
        e2_index = sentence.find("<e2>")

        if e1_index < e2_index:
            head_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            tail_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            head = "e1"
            tail = "e2"
        else:
            # print("e2")
            head_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            tail_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            head = "e2"
            tail = "e1"

        head_name = " ".join(head_name)
        tail_name = " ".join(tail_name)
        
        if prompt_type == "simple":
            prompt = semeval_prompt_template(sentence, relations, head, tail, head_name, tail_name)
        
        if prompt_type == "rag":
            prompt = semeval_prompt_template_rag(sentence, relations, head, tail, head_name, tail_name, context['similar_sentence'])
            
        prompts.append({"prompt":prompt, "relation":label})

    print("Number of Prompts:{0}".format(len(prompts)))

    return prompts
  

def generate_prompts(sentences, relations, similar_sentences,  dataset="tacred", prompt_type="rag"):
    """Regenerate the user query along with similar sentence.

    Args:
        sentences (list): list of sentences or dataset
        relations (list): list of relations
        similar_sentences (list): list of similar sentences
        dataset (str, optional): dataset name. Defaults to "tacred".
        prompt_type (str, optional): approach type. Defaults to "rag".

    Returns:
        list of prompts: list of regenerated prompts
    """

    prompts = []

    if dataset == "semeval":
        
        if prompt_type == "simple":
            prompts = semeval_format(sentences, relations, similar_sentences)
        else:
            prompts = semeval_format(sentences, relations, similar_sentences, prompt_type)
    else:
        normalized_sentences = [normalize_relation_example(item) for item in sentences]

        if prompt_type == "simple":
            prompts = tacred_format(normalized_sentences, relations, similar_sentences)
        else:
            prompts = tacred_format(normalized_sentences, relations, similar_sentences, prompt_type)
    
    return prompts
if  __name__ == "__main__":
    train_data = read_json("/Users/sefika/phd_projects/revision/RAG4RE/data/semeval/original_data/train_sentences.json")
    relation_data = read_json("/Users/sefika/phd_projects/revision/RAG4RE/data/semeval/original_data/train_relations.json")
    relation_names = read_json("/Users/sefika/phd_projects/revision/RAG4RE/data/semeval/original_data/relations.json")['relation']['names']
    train_relations = [relation_names[relation] for relation in relation_data]

    prompts = generate_prompts(train_data, train_relations, [],  dataset="semeval", prompt_type="simple")
    write_json( "/Users/sefika/phd_projects/revision/RAG4RE/data/semeval/original_data/train_prompts.json", prompts)
