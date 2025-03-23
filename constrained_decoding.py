from guidance import models, select

# %%capture
model = models.Transformers("gpt2")

result = model + f'Do you want a joke or a poem? A ' + select(['joke', 'poem'])

result = model + f'The sky is ' + select(['black', 'blue'])
