from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ## loading the model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# set model to eval (turn off dropout)
model.eval()  


# ## llm function

def run_llm(input_ids):
    with torch.no_grad():
        # run model
        outputs = model(input_ids)
        # get logits
        next_token_logits = outputs.logits[:, -1, :]
        # convert logits to probabilities using softmax
        probabilities = torch.softmax(next_token_logits, dim=-1)
        # Sample the next token
        return torch.multinomial(probabilities, num_samples=1)



# ## encode sentence

prompt = "The sky is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

input_ids

generated_ids = input_ids.clone()

# ## run llm

next_token = run_llm(generated_ids)
next_token

new_token = tokenizer.decode(next_token[0])
new_token

# concatenate
generated_ids = torch.cat((generated_ids, next_token), dim=1)

# Print the full generated text
full_text = tokenizer.decode(generated_ids[0])
print("\n\nFull generated text:")
print(full_text)
