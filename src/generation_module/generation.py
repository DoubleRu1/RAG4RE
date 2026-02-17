"""Generates responses for prompts using a language model defined in Hugging Face."""
"""Created by: Sefika"""


import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from transformers import T5Tokenizer, T5ForConditionalGeneration

class LLM(object):
    
    def __init__(self, model_id="google/flan-t5-xl"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-xl".
        """
        self.model_id = model_id
        self.maxmem = {"cpu": "300GB"}
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                with torch.cuda.device(i):
                    free_gb = int(torch.cuda.mem_get_info()[0] / 1024**3)
                self.maxmem[i] = f"{max(free_gb - 2, 1)}GB"

        if model_id=="google/flan-t5-xl":
            self.model, self.tokenizer = self.get_model(model_id)
        else: 
            self.model, self.tokenizer = self.get_model_decoder(model_id)
        
        
    def get_model(self, model_id="google/flan-t5-xl"):
        """_summary_

        Args:
            model_id (str, optional): LLM name at HuggingFace . Defaults to "google/flan-t5-xl".

        Returns:
            model: model from Hugging Face
            tokenizer: tokenizer of this model
        """
        tokenizer = T5Tokenizer.from_pretrained(model_id)
 
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = T5ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=False,
            torch_dtype=dtype,
            max_memory=self.maxmem,
        )
        return model,tokenizer
    
    def get_prediction(self, prompt, length=30):
        """_summary_

        Args:
            model : loaded model
            tokenizer: loaded tokenizer
            prompt (str): prompt to generate response 
            length (int, optional): Response length. Defaults to 30.

        Returns:
            response (str): response from the model
        """
        model_id_lower = self.model_id.lower()
        if "llama-3" in model_id_lower or "llama3" in model_id_lower:
            return self.get_prediction_llama3(prompt, length)
    
        model_device = self.model.device
        if model_device.type == "cpu" and torch.cuda.is_available():
            model_device = torch.device("cuda")
        inputs = self.tokenizer(prompt, add_special_tokens=True, max_length=526,return_tensors="pt").input_ids.to(model_device)
        
        outputs = self.model.generate(inputs, max_new_tokens=length)
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return response[0]


    def get_prediction_llama3(self, prompt, length=250, stype='greedy'):
        model = self.model
        tokenizer = self.tokenizer
        messages = [
        {"role": "system", "content": "You are a chatbot who extracts relations from text and returns only the relation label."},
        {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) # Ensure input_ids are on the correct device

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = model.generate(
            input_ids,
            max_new_tokens=length, # Use the passed length parameter
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        # Decode the generated response, excluding the input prompt tokens
        response = tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]


        return response

    
    def get_model_decoder(self, model_id="meta-llama/Llama-2-7b-chat-hf"):
        """loades the model from Hugging Face such llama and mistral

        Args:
            model_id (str, optional): _description_. Defaults to "meta-llama/Llama-2-7b-chat-hf".

        Returns:
            model: loaded model
            tokenizer: loaded tokenizer
        """

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=False,
            torch_dtype=dtype,
            max_memory=self.maxmem,
        )
        return model,tokenizer
