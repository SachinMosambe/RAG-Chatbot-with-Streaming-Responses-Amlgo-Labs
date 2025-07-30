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
        return f"""You are a helpful assistant. Use the context to answer the user question.
        
        Context: {context}
        Question: {query}
        """

    def generate(self, context, query, max_tokens=200):
        prompt = self.format_prompt(context, query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    