import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from PIL import Image
import io
import numpy as np
import open_clip
from sklearn.cluster import AgglomerativeClustering
import json
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BlipProcessor, BlipForConditionalGeneration



class ImageEmbedDataset(Dataset):
    def __init__(self, DATA_ROOT,image_paths, preprocess):
        self.data_root = DATA_ROOT
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        full_path = os.path.join(self.data_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        return self.preprocess(img), rel_path

class CaptionDataset(Dataset):
    def __init__(self, data_root, relative_paths):
        self.data_root = data_root
        self.relative_paths = relative_paths

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, idx):
        rel_path = self.relative_paths[idx]
        full_path = os.path.join(self.data_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        return img, rel_path


class CLIP_embedding:

	def __init__(self):
		self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
		self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
		self.model.eval() #disables things that are active during training cuz only using CLIP for inference
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = self.model.to(self.device)


	def preprocess_imgs(self, DATA_ROOT,all_image_paths):
		"""
		"""
		embed_dataset = ImageEmbedDataset(DATA_ROOT, all_image_paths, self.preprocess)
		embed_loader = DataLoader(embed_dataset, batch_size=64, shuffle=False, num_workers=4)

		return embed_loader

	def create_img_embeddings(self,DATA_ROOT, all_image_paths):
		"""
		img: an RGB img
		"""
		embed_loader = self.preprocess_imgs(DATA_ROOT, all_image_paths)
		
		all_embeddings  = []
		all_paths_ordered = []
		self.model.eval()

		with torch.no_grad():

			for images, paths in embed_loader:

				images = images.to(self.device)
				batch_embeddings =self.model.encode_image(images)

				#nrmalizing turns dot product into cosine similarity
				#Since, later on in the paper, to check similarity we use cosine similarity,
				#turning the embedding into unit vectors in the beginning lets us directly take the 
				#dot product of 2 embeddngs to get their cosine similarity.
				batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)  # normalize
				all_embeddings.append(batch_embeddings.cpu().numpy())
				all_paths_ordered.extend(paths)
		embeddings = np.concatenate(all_embeddings, axis=0)  # shape: [num_images, 512]
		return all_paths_ordered, embeddings

	def save_embeddings(self, paths, embeddings, name = "clip_embeddings"):
		"""
		"""
		np.savez(
    				name+".npz",
    				paths=np.array(paths),  # array of strings — perfectly valid
    				embeddings=embeddings              # array of floats — also valid
				)



	def create_text_embedding(self, text):
		"""
		text: a list of strings
		"""
		text = self.tokenizer(text).to(self.device)
		with torch.no_grad():
			text_embedding = model.encode_text(text) 
			text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # normalize

		return text_embedding




class Clustering_embeddings:

	def __init__(self, embeddings, clusters = None, distance_thld = 0.3, metric = "cosine", linkage = "average"):
		self.embeddings  = embeddings
		self.clustering = AgglomerativeClustering( n_clusters=clusters,              # threshold decides how many cluster
	    				  						   distance_threshold=distance_thld,   # paper uses similarity threshold δ=0.7. Distance = 1- similarity
	    				  						   metric=metric,
	    				  						   linkage=linkage)

	def create_Clusters(self):
		"""
		"""
		labels = self.clustering.fit_predict(self.embeddings)
		return labels

	def store_clusters(self,img_paths, labels, name = "clusters"):

		"""
		"""

		cluster_data = {}
		for c in set(labels):
			member_indices = [i for i,l in enumerate(labels) if l == c]
			mean_embedding = self.embeddings[member_indices].mean(axis = 0)
			member_paths = [img_paths[i] for i in member_indices]
			cluster_data[int(c)] = {"prototype_embedding": mean_embedding.tolist(),
							   "image_paths": member_paths}


		with open(name + ".json", "w") as f:
			json.dump(cluster_data, f)





class caption_with_BLIP:
		
	def __init__(self):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = self.model.to(self.device)

	def collate_fn(self,batch):
		images, paths = zip(*batch)
		return list(images), list(paths)

	def preprocess_images(self, DATA_ROOT, relative_paths):
		"""
		"""
		caption_dataset = CaptionDataset(DATA_ROOT, relative_paths)
		caption_loader = DataLoader(caption_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=self.collate_fn)
		return caption_loader

	def caption_imgs(self, DATA_ROOT, relative_paths):
		"""
		"""
		i = 0
		caption_loader = self.preprocess_images(DATA_ROOT, relative_paths)

		all_captions = []
		all_paths_ordered = []

		with torch.no_grad():
		    for images, paths in caption_loader:
		        i += 1
		        print(i)
		        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
		        outputs = self.model.generate(**inputs, max_new_tokens=30)
		        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)

		        all_captions.extend(captions)
		        all_paths_ordered.extend(paths)

		return all_paths_ordered, all_captions

	def save_captions(self, paths, captions, name = "captions"):
		"""
		"""
		paths_n_captions = {"paths":paths, "captions":captions}

		with open(name + ".json", "w") as f:
			json.dump(paths_n_captions, f)


DATA_ROOT = "/kaggle/input/datasets/hridayaannuncio24x7/emoset-118k"  # adjust to your actual mount path

with open(os.path.join(DATA_ROOT, "train.json"), "r") as f:
    entries = json.load(f)
subset = entries[:]
relative_paths = [entry[1] for entry in subset]  # relative paths like "image/amusement/amusement_12865.jpg"

"""
clip = CLIP_embedding()
#print(subset)
paths, embeddings = clip.create_img_embeddings(DATA_ROOT,relative_paths)
#subset = entries[:500]
clip.save_embeddings(paths, embeddings)
"""

blip = caption_with_BLIP()
paths, captions = blip.caption_imgs(DATA_ROOT, relative_paths)
blip.save_captions(paths, captions)