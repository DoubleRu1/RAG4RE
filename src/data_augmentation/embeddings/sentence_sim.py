import os
import sys
import json
import numpy as np
from numpy.linalg import norm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"
PROJECT_ROOT = os.path.dirname(PREFIX_PATH.rstrip("/"))


import configparser

def resolve_path(path):
    """Resolve configured paths against repository root when relative."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))

def read_json(path):
    """ Read json file"""
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_sentence_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item.get("tokens"), list):
        return " ".join([str(tok) for tok in item["tokens"]])
    if isinstance(item.get("tokens"), str):
        return item["tokens"]
    return item.get("sentence") or item.get("text") or item.get("sent") or ""


def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_sentence = get_sentence_text(train_line)
                
            context =  train_sentence
            train_similarities.append({"train":train_index, "simscore": sim, "sentence":context})

        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings for semeval dataset
    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings
    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """
    
    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_similarities.append({"train":train_index, "simscore":sim, "sentence":train_line})
        
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset="semeval"):
    """Compute similarity between test and train embeddings"""

    test_data = read_json(test_file)
    train_data = read_json(train_file)

    train_embeddings = np.load(train_emb)
    test_embeddings = np.load(test_emb)

    if dataset == "semeval":
        similarities = semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings)
    else:
        similarities = compute_similarity(test_data, train_data, train_embeddings, test_embeddings)

    write_json(output_sim_path, similarities)


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")

    test_file = resolve_path(config["SIMILARITY"]["test_file"])
    train_file = resolve_path(config["SIMILARITY"]["train_file"])
    train_emb = resolve_path(config["SIMILARITY"]["train_emb"])
    test_emb = resolve_path(config["SIMILARITY"]["test_emb"])
    output_sim_path = resolve_path(config["SIMILARITY"]["output_index"])
    dataset = config["SETTINGS"].get("dataset", "semeval")

    main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset)
