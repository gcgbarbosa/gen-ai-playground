import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# ## load model

model = api.load("word2vec-google-news-300")

len(model)

# ## getting embeddings

w1 = "princess"

w2 = "queen"

vec1 = model[w1]
vec2 = model[w2]

# ## calculating similarity

similarity = cosine_similarity([vec1], [vec2])
print(f"Cosine similarity between '{w1}' and '{w2}':", similarity[0][0])

similarity = cosine_similarity([vec1], [vec2])
print(f"Cosine similarity between '{w1}' and '{w2}':", similarity[0][0])
