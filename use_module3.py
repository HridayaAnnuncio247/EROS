import numpy as np
from Module3 import create_vocabulary
"""
data = np.load("clip_embeddings.npz")

# See what arrays are stored inside
print(len(data["embeddings"]))  # e.g. ['paths', 'embeddings']
embeddings = data["embeddings"]
paths = data["paths"]


emotion_embeddings = ["amusement","awe", "contentment","excitement","anger","disgust","fear", "sadness"]

for emo in emotion_embeddings:
	paths_emo = [p for p in paths if emo in p ]
	emb_emo = np.array([e for i,e in enumerate(embeddings) if emo in paths[i]])
	clustering = Clustering_embeddings(emb_emo)
	labels = clustering.create_Clusters()
	clustering.store_clusters(paths_emo, labels, emo+"clustering")

"""

import json
"""
# Open the file and load its content
with open('captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
#print(len(data["paths"]))

paths = data["paths"]
captions = data["captions"]
data = np.load("labels.npz")
labels = data["labels"]

print(labels[:5])
print(len(labels))
print(data.files)

with open('best_nouns.json', 'r', encoding='utf-8') as file:
	print("in best nouns json")
	data = json.load(file)
#print(data["paths"][:] == paths)
nouns = data["best_nouns"]

n = create_vocabulary(paths[:], labels[:], captions[:])
s_nouns = n.convert_to_singular(nouns)
paths_n_captions = {"paths":paths, "singular_nouns":s_nouns, "labels":labels.tolist()}

with open("singular_nouns.json", "w") as f:
	json.dump(paths_n_captions, f)

n = create_vocabulary(None, None, None)

with open('singular_nouns.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
print(len(data["singular_nouns"]))
s_nouns = data["singular_nouns"]

labels = data["labels"]
S0, S1, Sn = n.pos_neg_vocab( s_nouns, labels)

print(len(S0), len(S1), len(Sn))
vocab = {"positive":list(S1), "negative":list(S0), "neutral":list(Sn)}

with open("vocabulary.json", "w") as f:
	json.dump(vocab, f)


with open('vocabulary.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
print(data["positive"][:10])
print(data["negative"][:10])
print(len(data["positive"]))
print(len(data["negative"]))
"""
"""
with open('best_nouns.json', 'r', encoding='utf-8') as file:
	print("in best nouns json")
	data = json.load(file)
#print(data["paths"][:] == paths)
nouns = data["best_nouns"]


with open('singular_nouns.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
s_nouns = data["singular_nouns"]

print(nouns[:50],"/n", s_nouns[:50], len(nouns), len(s_nouns))
"""