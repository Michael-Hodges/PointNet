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
		self.pointfiles = os.path.join(self.path, 'train_data/')#, ''.format(i)) for i in os.path.join(self.path, train_data))]
		# Fill dict self.classes with {class_number, class_name}
		self.source = []
		self.label = {}
		with open(self.classfile, 'r') as f:
			for i, line in enumerate(f):
				line = line.strip().split()
				self.classes[line[1]] = line[0]
				self.label[line[0]] = i
		print(self.label)
		# print(self.classes)
		
		# data comes in as shapenet/numbers/collection.pts
		# we need to load data from each folder and label each point per folder
		self.points = [os.path.join(self.pointfiles, x) for x in self.classes] # self.points has each class folder path
		self.data_paths = []
		for i in self.points:
			for k in os.listdir(i):
				self.data_paths.append(os.path.join(i,k)) # datapaths contain path to each point


	def __getitem__(self, index):
		_,trn_tst, cat, pt = self.data_paths[index].strip().split('/') # splitting datapath to get
		pts = np.loadtxt(self.data_paths[index], dtype=np.float32) # loads points as float32
		points_sel = np.random.choice(range(0,len(pts)-1), size=2000, replace=True) # All batches need to have same number of points. This will sample from all the points uniformly and keep them all equal size
		pts = pts[points_sel,:]
		pts = torch.from_numpy(pts)
		item_class = torch.from_numpy(np.array([self.label[self.classes[cat]]]))
		# TODO: center all points and and normalize
		return pts, item_class # returns set of points and class num [0-15]

	def __len__(self):
		return len(self.data_paths)


if __name__ == '__main__':
	classify_net = ShapeNetClassify('ShapeNet')
	points, item_class = classify_net.__getitem__(50)
	class_len = classify_net.__len__()

	print("current class: {}".format(item_class.item()))
	print("class length: {}".format(class_len))
	print("points: {}".format(points))
	loader = data.DataLoader(classify_net, batch_size = 16, shuffle=True, drop_last=True)
	
	for batch_num, (sample, label) in enumerate(loader):
		print("{}: {}".format(batch_num, sample))
















