import os
import sys

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from refinement import postprocessing
from data_augmentation.prompt_generation.prompt_generation import generate_prompts, normalize_relation_example
from generation_module.generation import LLM
import configparser
from utils import read_json, write_json
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

def benchmark_data_augmentation_call(config_file_path):
    """
    This function is used to benchmark the retrieval module.
    Args:
    config_file_path: str: Path to the config file.
    """
    print("PREFIX_PATH", PREFIX_PATH)

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH + config_file_path)
    
    test_data_path = config["PATH"]["test_data_path"]
    similar_sentences_path = config["SIMILARITY"]["output_index"]
    relations_path = config["PATH"]["relations_path"]
    dataset = config["SETTINGS"]["dataset"]
    prompt_type = config["SETTINGS"]["prompt_type"]
    model_name = config["SETTINGS"]["model_name"]
    similar_sentences = read_json(similar_sentences_path)
    relations = read_json(relations_path)

    if dataset != "semeval":
        if isinstance(relations, dict):
            relation_list = list(relations.keys())
        elif isinstance(relations, list):
            relation_list = [str(item) for item in relations]
        else:
            raise ValueError("relations_path must point to a json/jsonl dict or list for non-semeval datasets.")
    else:
        relation_list = relations
    test_data = read_json(test_data_path)
    if dataset != "semeval":
        test_data = [normalize_relation_example(item) for item in test_data]

    if prompt_type == "rag":
        # print("RAG")
        output_prompts_path = config["OUTPUT"]["rag_test_prompts_path"]
        output_responses_path = config["OUTPUT"]["rag_test_responses_path"]
        prompts = generate_prompts(test_data, relation_list, similar_sentences,  dataset, prompt_type)
    else:
        output_prompts_path = config["OUTPUT"]["simple_prompt_path"]
        output_responses_path = config["OUTPUT"]["simple_prompt_responses_path"]
        prompts = generate_prompts(test_data, relation_list, similar_sentences,  dataset, prompt_type)

    llm_instance = LLM(model_name)
    
    responses = []

    for prompt in prompts:
        prompt = prompt["prompt"]

        if "t5" not in model_name.lower():
            prompt = f"[INST]{prompt}[/INST] Answer:"

        response = llm_instance.get_prediction(prompt)
        responses.append(response)

    responses = postprocessing(dataset, test_data, responses, relation_list, model_name)
    
    write_json(output_prompts_path, prompts)
    write_json(output_responses_path, responses)
    
