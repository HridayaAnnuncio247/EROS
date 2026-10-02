from Module2 import Module2
from Module4 import Module4
import zipfile
import io
from PIL import Image
import json

with open('captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


path = data["paths"][0]
caption = data["captions"][0]

with zipfile.ZipFile("C:/Users/My Document/Documents/00EROSdataset/EmoSet-118K.zip") as z:
    with z.open(path) as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")

print(path,caption)
img.show()


m2 = Module2("resnet18_emoset_binary.pt")
Mfinal = m2.CAM_and_SAM(img, 0.5)

m4 = Module4()

prompt = "a young girl sitting at a table with a vase of flowers"

new_img = m4.generate(img, Mfinal, prompt)
new_img.show()


