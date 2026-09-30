import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from PIL import Image
import io
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import os
#from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from transformers import pipeline

import spacy
import random
import time


class EmoTree:

	def __init__(self):
		"""
		"""
		self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
		#self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		#self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.model = self.model.to(self.device)

		#self.nlp = spacy.load("en_core_web_sm")
		#self.img_paths = img_paths
		#self.labels = labels
		#self.captions = captions
		self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")




	def best_candidate_word(self, all_nouns, captions, label):
		"""
		"""
		best_noun = []
		for caption, nouns in zip(captions, all_nouns):
			if not nouns:  # edge case: caption had no nouns at all
				best_noun.append(None)
				continue

			result = self.classifier(
	            caption,
	            candidate_labels=nouns,
	            hypothesis_template=f" {{}} causes {label} emotion to the text and is semantically coherent with the text."
	        )
			#print(caption,result)
			best_noun.append(result["labels"][0])

		return best_noun

	def sample_best_noun(self, vocab, captions, sample_size = 50):
		"""
		dict vocab: Has positive and negative vocabulary. The keys are "positive" and "negative". The corresponding values are lists. 
		"""
		
		all_nouns = {"positive":[], "negative":[]}

		for emo in ["positive", "negative"]:
			for cap in captions:
				sample_nouns = random.sample(vocab[emo], sample_size)
				all_nouns[emo].append(sample_nouns)

		all_best_positive_nouns = self.best_candidate_word(all_nouns["positive"], captions, "positive")
		all_best_negative_nouns = self.best_candidate_word(all_nouns["negative"], captions, "negative")

		return all_best_positive_nouns, all_best_negative_nouns


	def build_motifs(self, caption,target_emotion , candidate_noun, max_new_tokens=100):
		"""
		"""
		prompt = "given the source image description " +  caption + ", the target emotion " + target_emotion + " and the candidate concept " + candidate_noun + "revise the description by replacing or modifying its emotion-related concept with a more specific target concept. Generate a realistic visual motif by specifying the target concept, its attributes, related actions, and scene context." 
		messages = [{"role": "user", "content": prompt}]
		inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    
		with torch.no_grad():
			outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
    
		response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
		return response

		
e = EmoTree()
with open('vocabulary.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
S1 = data["positive"]
S0 = data["negative"]
dic = {"positive":S1, "negative":S0}

with open('captions.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
captions = data["captions"][:20]

start_time = time.perf_counter()

pos, neg = e.sample_best_noun(dic, captions,50)

end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")


target_noun = {"positive": pos, "negative": neg}

with open("target_noun.json", "w") as f:
	json.dump(target_noun, f)
