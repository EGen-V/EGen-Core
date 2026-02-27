import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_inference():
    model_id = "ErebusTN/EGen-SA1Q9"
    print(f"Loading model and tokenizer: {model_id}...")
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check if GPU is available to set the correct dtype and device map
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds on device map: {device_map}")

    prompt = "Explain the significance of the Athena Project in 2025."
    print(f"\nPrompt: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    start_time = time.time()
    print("Generating response...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    gen_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Output ---\n")
    print(response)
    print(f"\n--------------\n")
    print(f"Generation completed in {gen_time:.2f} seconds.")

if __name__ == "__main__":
    test_inference()
