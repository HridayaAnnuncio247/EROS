import numpy as np

data = np.load("clip_embeddings.npz")

# See what arrays are stored inside
print(len(data["embeddings"]))  # e.g. ['paths', 'embeddings']


import json

# Open the file and load its content
with open('captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data["paths"]))