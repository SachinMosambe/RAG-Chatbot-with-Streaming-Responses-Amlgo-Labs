from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
os.makedirs("./offload", exist_ok=True)


class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float32,      
            device_map="auto",               
            offload_folder="./offload"
        )

    def format_prompt(self, context, query):
        return  """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
answer:
"""

    def generate(self, context, query, max_tokens=200):
        prompt = self.format_prompt(context, query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
