
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # set model to evaluation mode

# Starting prompt
prompt = "In a distant future,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate tokens one by one
generated_ids = input_ids.clone()
num_tokens_to_generate = 20  # for example

print("\nToken-by-Token Generation:")
with torch.no_grad():
    for _ in range(num_tokens_to_generate):
        # Get the model outputs for the current sequence
        outputs = model(generated_ids)
        # Focus on the logits for the last token
        next_token_logits = outputs.logits[:, -1, :]
        # Convert logits to probabilities
        probabilities = torch.softmax(next_token_logits, dim=-1)
        # Sample the next token
        next_token = torch.multinomial(probabilities, num_samples=1)
        # Append the token to the sequence
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        # Decode and display the newly generated token
        new_token = tokenizer.decode(next_token[0])
        print(new_token, end="")

# Print the full generated text
full_text = tokenizer.decode(generated_ids[0])
print("\n\nFull generated text:")
print(full_text)
