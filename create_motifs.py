from EmoTree import EmoTree

e = EmoTree()
with open('vocabulary.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
S1 = data["positive"]
S0 = data["negative"]
dic = {"positive":S1, "negative":S0}

with open('captions.json', 'r', encoding='utf-8') as file:
	data = json.load(file)
captions = data["captions"][:10]
paths = data["paths"]

data = np.load("labels.npz")
labels = data["labels"]



for target_emotion in ["positive", "negative"]:

	response = build_motifs( captions[i],target_emotion , candidate_noun[i], max_new_tokens=100)		