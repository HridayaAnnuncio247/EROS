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