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

class dataset_emo:
    """
    This class is used to load EmoSet images preprocessed.
    """

    def __init__(self, zip_path, json_path, preprocess):
        """
        str zip_path: The zip file where all the images/annotations etc are stored. 
        str json_path: this is the path to the json folder that has the list of image paths and labels.
                  It can be a train.json, val.json or test.json.
        preprocess: A function that transforms each image
        """

        # converting the 8 emotions to just 2 emotions: 1 is positive and 0 is negative
        self.label_dict = {
        "amusement":1, 
        "awe":1, 
        "contentment": 1,
        "excitement": 1,
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "sadness": 0}


        self.zip_path = zip_path
        self.json_path = json_path
        self.preprocess = preprocess
        full_path = os.path.join(self.zip_path, self.json_path)
        with open(full_path, "r") as f:
            data_entries = json.load(f)
        
        self.path_and_labels = []

        for i in data_entries:
            self.path_and_labels += [[i[1], self.label_dict[i[0]]]]


        print(self.path_and_labels[0:5])

    def __len__(self):
        return len(self.path_and_labels)


    def __getitem__(self, img_entry_index):
        entry = self.path_and_labels[img_entry_index]
        img_path = entry[0]
        img_label = entry[1]
        full_path = os.path.join(self.zip_path, img_path)
        img = Image.open(full_path).convert("RGB")
        transformed_img = self.preprocess(img)
        return transformed_img, img_label



"""with zipfile.ZipFile("EmoSet-118K.zip") as z:
    names = z.namelist()
    folders = []
    for name in names:
        folder = name.split("/")[0:2]
        if folder not in folders:
            folders.append(folder) 
    print(folders)
"""

#generally images have RGB values between 0 and 255. However, to enter resnet18 it they have t o be between 0 and 1
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean, what ResNet18 was pretrained on
        std=[0.229, 0.224, 0.225],   # ImageNet std deviation
    ),
])


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

#converting the last fc layer to a binary output instead of a 1000 class softmax
#that was originally there in ResNet18.
#This is the only fc layer in the entire model
model.fc = nn.Linear(in_features=512, out_features=2)#since crossentropy loss, output has to have 2 neurons (1 not enough)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr =1e-4 )


train_dataset = dataset_emo("/kaggle/input/datasets/hridayaannuncio24x7/emoset-118k", "train.json", preprocess)
train_data_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 4)

val_dataset = dataset_emo("/kaggle/input/datasets/hridayaannuncio24x7/emoset-118k", "val.json", preprocess)
val_data_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True, num_workers = 4)


# --- Validation function ---
def evaluate(model, data_loader, criterion, device):
    model.eval()  # switch to eval mode (disables dropout, uses running stats for batchnorm)
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # don't track gradients — saves memory/time, not needed for eval
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)  # pick highest-scoring class per image
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    model.train()  # switch back to train mode before returning to the training loop
    return avg_loss, accuracy



epochs_without_improvement = 0
epochs = 100
model.train()
max_no_improvement_epochs = 3
min_delta = 1e-3
best_loss = float("inf")
for e in range(epochs):
    total_loss = 0
    print("epoch", e+1)
    #b = 0
    for images, labels in train_data_loader: #gets a batch of 32 each iteration
        #b += 1
        #print("batch", b)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) #shape: [batch_size (here 32), 1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_data_loader)
    print("Epoch", e + 1, "loss:", avg_loss )
    val_loss, val_acc = evaluate(model, val_data_loader, criterion, device)
    print(f"Epoch {e+1} | Train loss: {avg_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

    if best_loss - val_loss > min_delta:
        best_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "resnet18_emoset_binary.pt")  # save the best model
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= max_no_improvement_epochs:
        print("No improvement for", max_no_improvement_epochs,"epochs — stopping early at epoch {e+1}")
        break
#torch.save(model.state_dict(), "resnet18_emoset_binary.pt")
#img_transformed = preprocess(img)
#print(preprocess)
#print(preprocess.mean)
#print(preprocess.std)
#model.train()
#print(model)

