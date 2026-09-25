import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import zipfile
import json
from PIL import Image
import io
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import sam_model_registry, SamPredictor
import numpy as np


class Module2:
	def __init__(self, state_dict):
		"""
		state_dict: the model with weights for resnet18 architecture that has to be loaded.
		"""
		self.model = resnet18(weights=None)  # architecture only, no pretrained weights this time
		self.model.fc = nn.Linear(in_features=512, out_features=2)  # rebuild your custom head first
		self.model.load_state_dict(torch.load(state_dict))
		target_layers = [self.model.layer4[-1]]
		self.cam = GradCAM(model= self.model, target_layers=target_layers)
		
		sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    	sam.to(device="cpu")
		self.sam_predictor = SamPredictor(sam)

	def CAM_preprocess(self, img):
		"""
		"""
		preprocess = transforms.Compose([
	    transforms.Resize((512, 512)),
	    transforms.ToTensor(),
	    transforms.Normalize(
	        mean=[0.485, 0.456, 0.406],  # ImageNet mean, what ResNet18 was pretrained on
	        std=[0.229, 0.224, 0.225],   # ImageNet std deviation
	    ),
			])
		transformed_img = preprocess(img)
		return transformed_img

	def SAM_preprocess(self, img):
		"""
		"""
		preprocess = transforms.Compose([
	    transforms.Resize((512, 512)),])
		transformed_img = preprocess(img)
		return transformed_img


	def CAM_mask(self, img):
		"""
		img : transformed image to be processed.
		"""
		grayscale_cam = self.cam(input_tensor = img, targets = None)
		heatmap = grayscale_cam[0] #all steps of grad cam are done internally. How cool!
		return heatmap

	def threshold_CAM(self, M, tau):
		"""
		M : averaged heatmap
		tau: the threshold pixel value above which pixel values should be changed to 1. 
		"""
		Mcam = (M>tau).astype(np.float32)
		return Mcam

	def SAM_peak_activation(self, M):
		"""

		"""
		peak_index = np.unravel_index(np.argmax(M), M.shape)  # (row, col) = (i*, j*)
		i_star, j_star = peak_index
	    # SAM expects (x, y) = (col, row)
		point_coords = np.array([[j_star, i_star]])#creating a 2d array with shape 1X2
		point_labels = np.array([1])  # 1 = foreground point thus segment the object that has this point
	    #this is basically the poit where the emotion relevant region has peaked
		return point_coords,point_labels

	def SAM_mask(self,img,M):
		"""
		"""
		img_array = np.array(img) #SAM expects input to be an array

		#set_image is SAM's encoder -> the whole image is converted to an embedding. This is the most expensie part.
		#In case one wants to segment different parts of the images,this command is separated from predict() so that it only
		#has to be run once
		self.sam_predictor.set_image(img_array)

		point_coords,point_labels = self.SAM_peak_activation(M)

		
		#Since a single point can be ambiguous wrt which object it represents(a small part of a bigger object or the bigger object itself), 
		#we have made multimask output TRUE.
		#THis means that SAM will give its 3 candidate masks.
		#SAM works differently when it gives 3 candidate masks vs just 1
		# The top scored candidate mask may not be the same as the "one" mask it gives when multimask is FALSE
		masks, scores, _ = self.sam_predictor.predict(
	    				   point_coords=point_coords,
	                       point_labels=point_labels,
	    		           multimask_output=True,
						   )

		#chooses mask with highest score
		Msam = masks[np.argmax(scores)].astype(np.float32)

		return Msam

	def CAM_and_SAM(self, img, tau):
		"""
		"""

		cam_img = self.CAM_preprocess(img)
		sam_img = self.SAM_preprocess(img)

		M = self.CAM_mask(cam_img)
		Mcam = self.threshold_CAM(M, tau)
		Msam = self.SAM_mask(sam_img,M)	

		#Each pixel of both masks compared. If even one of them has the value 1, keep that pixel in Mfinal
		Mfinal = np.maximum(Mcam, Msam)
		
		return Mfinal
