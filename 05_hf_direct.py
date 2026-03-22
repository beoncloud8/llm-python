from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# Load tokenizer only (since PyTorch is not available for Python 3.13)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define prompt template function
def generate_response(profession):
    prompt = f"You had one job 😡! You're the {profession} and you didn't have to be sarcastic"
    
    # Tokenize the prompt to show it works (without PyTorch tensors)
    inputs = tokenizer(prompt)
    
    # Since we can't generate without PyTorch, just return the prompt
    return f"Prompt: {prompt}\nTokenized length: {len(inputs['input_ids'])} tokens"

# Test with different professions
print(generate_response("customer service agent"))
print(generate_response("politician"))
print(generate_response("Fintech CEO"))
print(generate_response("insurance agent"))