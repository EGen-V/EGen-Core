import sys
import os
import logging
import time

# Give us debug statements
logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.abspath('./egen_core'))
from egen_core import AutoModel
import torch

def run_test():
    model_id = "ErebusTN/EGen-SA1Q9"
    print(f"Loading EGen-Core AutoModel for {model_id}...")
    start = time.time()
    
    try:
        model = AutoModel.from_pretrained(model_id)
        print(f"Model initialized in {time.time() - start:.2f}s")
        
        prompt = "Explain the significance of the Athena Project in 2025."
        inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Generating response...")
        start = time.time()
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            use_cache=True,
            return_dict_in_generate=True
        )
        print(f"Generated in {time.time() - start:.2f}s")
        response = model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print("\n--- Output ---\n")
        print(response)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_test()
