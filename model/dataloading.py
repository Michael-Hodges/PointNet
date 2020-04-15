import os
import os.path
import torch
import torch.utils.data as data
import numpy as np


class ShapeNetClassify(data.Dataset):
	def __init__(self, path):
		self.path = path  # path to data folder of shapenet
		self.classes = {} # dict of the form {class_number, class_name}
		self.classfile = os.path.join(self.path, 'class_labels.txt')  # create path to file that contains class numbners and labels
		with open(self.classfile, 'r') as f:
			for line in f:
				line = line.strip().split()
				self.classes[line[1]] = line[0]
		print(self.classes)
		for v, k in self.classes.items():
			print(v, k)
		

	def __getitem__(self, index):
		pass

	def __len__():
		pass


if __name__ == '__main__':
	classify_net = ShapeNetClassify('ShapeNet')