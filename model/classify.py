import os
import numpy as np
import pptk
import argparse
import torch

import model
import dataloading
from train import DEVICE


def visualize(point_data):
	x = np.loadtxt(point_data, dtype=np.float32)
	v = pptk.viewer(x)
	v.set(point_size=0.001)
	input("Press enter to continue...")

def load_model(model_type, model_path):
	if model_type == 'Vanilla_Classify_Net':
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		classifier.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
	return classifier

def classify(model, point_data):
	model = model
	data_input = np.loadtxt(point_data, dtype=np.float32)

	data_tensor = torch.tensor(data_input, device=DEVICE).unsqueeze(0)
	data_tensor = data_tensor.permute(0,2,1)
	model.eval()
	with torch.no_grad():
		output = model(data_tensor)
	
	_, class_number = torch.max(output[0], dim=0)
	# print(class_number)
	class_dict = load_class_dict(point_data)
	print("Model Class: {}".format(class_dict[class_number.item()]))

def load_class_dict(path):
	class_dict = {}
	root_path = path.strip().split('/')
	# print(root_path)
	root_path = root_path[0]
	class_file = os.path.join(root_path, 'class_labels.txt')
	with open(class_file, 'r') as f:
		for i, line in enumerate(f):
			line = line.strip().split()
			class_dict[i] = line[0]
	# print(class_dict)
	return class_dict

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', default ='ShapeNet/test_data/02691156/016871.pts', type=str)
	parser.add_argument('--model', default ='Vanilla_Classify_Net', type=str)
	parser.add_argument('--load', default= 'model/pn_classify.pt', type=str)
	args = parser.parse_args()
	
	classifier = load_model(args.model, args.load)
	visualize(args.path)
	classify(classifier, args.path)