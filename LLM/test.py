import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose the model you want to load
model_name = "gpt2"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prompt
prompt = "electricity is"

# Tokenize and encode the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    **inputs,
    max_length=50,       # Adjust max_length based on how much text you want to generate
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,     # Lower for more focused output, higher for more randomness
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)