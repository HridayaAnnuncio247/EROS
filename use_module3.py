import numpy as np
from Module3 import Clustering_embeddings

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





