import os
import numpy as np
import pptk
import argparse
import torch

import model
import dataloading
from train import DEVICE


def visualize(point_data, action, *args):
	if action == 'classify':
		x = np.loadtxt(point_data, dtype=np.float32)
		v = pptk.viewer(x)
		v.set(point_size=0.001)
		print("Press enter on visual to continue...")
		v.wait()
		v.close()
	if action == 'segment':
		x = np.loadtxt(point_data, dtype=np.float32)
		# v = pptk.viewer(x)
		labels = args[0]
		loc = []
		for i in range(6):
			loc.append(np.argwhere(labels ==i))
			# print("Label: {}, Locs: {}".format(i, loc))
		v = pptk.viewer(x)
		v.attributes(labels)
		# v.color_map('cool', scale=[0,5])
		v.set(point_size=0.001)
		print("Press enter on visual to continue...")
		v.wait()
		v.close()


def load_model(model_type, model_path):
	if model_type == 'Vanilla_Classify_Net':
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		classifier.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
	if model_type == 'Vanilla_Segment_Net':
		classifier = model.Vanilla_Segment_Net(output_dim=6)
		classifier.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
	return classifier


def classify(model, point_data, action):
	model = model.to(DEVICE)

	data_input = np.loadtxt(point_data, dtype=np.float32)
	if action=='classify':
		visualize(point_data, action)
		data_tensor = torch.tensor(data_input, device=DEVICE).unsqueeze(0)
		# data_tensor = data_tensor.permute(0,2,1)
		model.eval()
		with torch.no_grad():
			output = model(data_tensor)
		
		_, class_number = torch.max(output[0], dim=0)
		# print(class_number)
		class_dict = load_class_dict(point_data)
		print("Model Class: {}".format(class_dict[class_number.item()]))
	if action == 'segment':
		data_tensor = torch.tensor(data_input, device=DEVICE).unsqueeze(0)
		# data_tensor = data_tensor.permute(0,2,1)
		model.eval()
		with torch.no_grad():
			output = model(data_tensor)
		_, label = torch.max(output[0], dim=0)
		visualize(point_data, action, label.data.numpy())


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
	parser.add_argument('--action', default='classify', type=str)
	args = parser.parse_args()
	
	classifier = load_model(args.model, args.load)
	
	classify(classifier, args.path, args.action)