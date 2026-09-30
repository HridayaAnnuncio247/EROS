import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import zipfile
import json
from PIL import Image
import io
import os
import numpy as np


label_dict = {
        "amusement":1, 
        "awe":1, 
        "contentment": 1,
        "excitement": 1,
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "sadness": 0}


 
def save_embeddings( paths, labels, name = "labels"):
		"""
		"""
		np.savez(
    				name+".npz",
    				paths=np.array(paths),  # array of strings — perfectly valid
    				labels=labels              # array of floats — also valid
				)


with open('captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

ordered_paths = []
labels = []
for i in data["paths"]:
	for emo in label_dict:
		if emo in i:
			labels.append(label_dict[emo])
			ordered_paths.append(i)
			break

save_embeddings(ordered_paths, labels)
