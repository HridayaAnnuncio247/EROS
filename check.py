import numpy as np

#data = np.load("clip_embeddings.npz")

# See what arrays are stored inside
#print(len(data["embeddings"]))  # e.g. ['paths', 'embeddings']


import json
from Module3 import create_vocabulary

with open('captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
#print(len(data["paths"]))

paths = data["paths"]
captions = data["captions"]
data = np.load("labels.npz")
labels = data["labels"]
n = create_vocabulary(paths[:], labels[:], captions[:])
nouns = n.extract_nouns()
best_nouns = n.most_related_noun(nouns)
n.save_nouns(nouns, "all_nouns")

n.save_nouns(best_nouns, "best_nouns1")




with open('target_noun.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
p_nouns = data["positive"]

with open('target_prompts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
prompts = data["prompts"]

with open('best_nouns1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data.keys())
paths1 = data["paths"]
print(paths[:] == paths1[:])

best_nouns = data["best_nouns"]

for i in range(10):
	print("caption:",captions[i])
	print("All nouns", nouns[i])
	print("best noun:",best_nouns[i])
"""

for i in range(10):
	print("caption:",captions[i])
	print("target noun:",nouns[i])
	print("motif:",prompts[i])
	print()

print("negative")

for i in range(10):
	print("caption:",captions[i])
	print("target noun:",nouns[i])
	print("motif:",prompts[i+20])
	print()

"""