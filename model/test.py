import os

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

import model
import dataloading
from train import DEVICE
from classify import load_class_dict

def load_model(model_type, model_path):
	if model_type == 'Vanilla_Classify_Net':
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		classifier.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
	return classifier

def counter(path):
	cur_path = os.path.join(path, 'test_data/04379243/')
	print(len(os.listdir(cur_path)))
# Name            ID          Val_samples
# Airplane        02691156    341
# Bag             02773838    14
# Cap             02954340    11
# Car             02958343    158
# Chair           03001627    704
# Earphone        03261776    14
# Guitar          03467517    159
# Knife           03624134    80
# Lamp            03636649    286
# Laptop          03642806    83
# Motorbike       03790512    51
# Mug             03797390    38
# Pistol          03948459    44
# Rocket          04099429    12
# Skateboard      04225987    31
# Table           04379243    848
def test(model, path, act, dataset):
	classifier = model
	if act == 'classify':
		if dataset == 'shapenet':
			test_data = dataloading.ShapeNetClassify(path, 'test')
		
		test_loader = data.DataLoader(dataset=test_data, batch_size=16, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=False, timeout=0,
			worker_init_fn=None)
		loss_func = nn.CrossEntropyLoss()
		running_total_correct = np.zeros(16)
		running_total = np.zeros(16)

		with torch.no_grad():
			for _, (inputs, labels) in enumerate(test_loader):
				classifier.eval()
				inputs = inputs.permute(0,2,1)
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = classifier(inputs)
				_, predicted = torch.max(outputs.data, 1)
				correct = (predicted == labels).squeeze()
				for i in range(len(labels)):
					label = labels[i]
					running_total_correct[label] += correct[i]
					running_total[label] += 1

		class_dict = load_class_dict(path)		
		print("Per Class Accuracy:")
		print("-------------------------")
		for i, (correct, samples) in enumerate(zip(running_total_correct, running_total)):
			print("{0:>10}: {1:2.2f}".format(class_dict[i], 100*correct/samples))
		print("-------------------------")
		print("Total Accuracy: {0:2.2f}".format(100*np.sum(running_total_correct/np.sum(running_total))))

	if act == 'segment':
		if dataset == 'shapenet':
			test_data = dataloading.ShapeNetSegment(path, 'val')
		
		test_loader = data.DataLoader(dataset=test_data, batch_size=16, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=False, timeout=0,
			worker_init_fn=None)
		loss_func = nn.CrossEntropyLoss()
		running_total_correct = np.zeros(16)
		running_total = np.zeros(16)

		with torch.no_grad():
			for _, (inputs, labels) in enumerate(test_loader):
				classifier.eval()
				inputs = inputs.permute(0,2,1)
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = classifier(inputs)
				_, predicted = torch.max(outputs.data, 1)
				correct = (predicted == labels).squeeze()
				for i in range(len(labels)):
					label = labels[i]
					running_total_correct[label] += correct[i]
					running_total[label] += 1

		class_dict = load_class_dict(path)		
		print("Per Class Accuracy:")
		print("-------------------------")
		for i, (correct, samples) in enumerate(zip(running_total_correct, running_total)):
			print("{0:>10}: {1:2.2f}".format(class_dict[i], 100*correct/samples))
		print("-------------------------")
		print("Total Accuracy: {0:2.2f}".format(100*np.sum(running_total_correct/np.sum(running_total))))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='shapenet', type=str)
	parser.add_argument('--action', default='classify', type=str)
	parser.add_argument('--path', default='ShapeNet', type=str)
	parser.add_argument('--model', default ='Vanilla_Classify_Net', type=str)
	parser.add_argument('--load', default= 'model/pn_classify.pt', type=str)
	args = parser.parse_args()
	classifier = load_model(args.model, args.load)
	test(classifier, args.path, args.action, args.dataset)
	# counter(args.path)