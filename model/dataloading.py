import os
import os.path
import torch
import torch.utils.data as data
import numpy as np


# data folder should be set up as follows:
# root path named by user (default ShapeNet)
# subfolders class_labels.txt converts folder numbers to class labels
# subfolders cont. test_data/ train_data/ val_data/ train_label/ val_label/
# Airplane        02691156
# Bag             02773838
# Cap             02954340
# Car             02958343
# Chair           03001627
# Earphone        03261776
# Guitar          03467517
# Knife           03624134
# Lamp            03636649
# Laptop          03642806
# Motorbike       03790512
# Mug             03797390
# Pistol          03948459
# Rocket          04099429
# Skateboard      04225987
# Table           04379243

class ShapeNetClassify(data.Dataset):
	def __init__(self, path, train_val_test):
		self.path = path  # path to data folder of shapenet
		self.classes = {} # dict of the form {class_number, class_name}
		self.classfile = os.path.join(self.path, 'class_labels.txt')  # create path to file that contains class numbners and labels
		if train_val_test == 'train':
			self.pointfiles = os.path.join(self.path, 'train_data')#, ''.format(i)) for i in os.path.join(self.path, train_data))]
		if train_val_test == 'val':
			self.pointfiles = os.path.join(self.path, 'val_data')
		if train_val_test == 'test':
			self.pointfiles = os.path.join(self.path, 'test_data' )
		# Fill dict self.classes with {class_number, class_name}
		self.source = []
		self.label = {}
		with open(self.classfile, 'r') as f:
			for i, line in enumerate(f):
				line = line.strip().split()
				self.classes[line[1]] = line[0]
				self.label[line[0]] = i
		# print(self.label)
		#{'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}
		# print(self.classes
		# {'02691156': 'Airplane', '02773838': 'Bag', '02954340': 'Cap', '02958343': 'Car', '03001627': 'Chair', '03261776': 'Earphone', '03467517': 'Guitar', '03624134': 'Knife', '03636649': 'Lamp', '03642806': 'Laptop', '03790512': 'Motorbike', '03797390': 'Mug', '03948459': 'Pistol', '04099429': 'Rocket', '04225987': 'Skateboard', '04379243': 'Table'}

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
		# print("from data_loader: {}".format(item_class.item()))
		# TODO: center all points and and normalize
		return pts, item_class.item() # returns set of points and class num [0-15]

	def __len__(self):
		return len(self.data_paths)

class ShapeNetSegment(data.Dataset):
	def __init__(self, path, train_val_test):

		self.path = path
		self.classes = {}
		self.classfile = os.path.join(self.path, 'class_labels.txt')
		if train_val_test == 'train':

			self.pointfiles = os.path.join(self.path, 'train_data')
			self.labelfiles = os.path.join(self.path, 'train_label')
		if train_val_test == 'val':
			self.pointfiles = os.path.join(self.path, 'val_data')
			self.labelfiles = os.path.join(self.path, 'val_label')
		if train_val_test == 'test':
			print("No test labels provided")
			self.pointfiles = os.path.join(self.path, 'test_data')
		self.source = []
		self.label = {}
		with open(self.classfile, 'r') as f:
			for i, line in enumerate(f):
				line = line.strip().split()
				self.classes[line[1]] = line[0]
				self.label[line[0]] = i
		self.points = [os.path.join(self.pointfiles, x) for x in self.classes] # self.points has each class folder path
		self.data_paths = []
		for i in self.points:
			for k in os.listdir(i):
				self.data_paths.append(os.path.join(i,k)) # datapaths contain path to each point


	def __getitem__(self, index):
		
		# print(self.data_paths[index])
		# next 2 lines just get the path to the proper label item
		_path, trn_tst, c_class, c_item = self.data_paths[index].strip().split('/')
		label_path = os.path.join(self.path, (trn_tst.split('_')[0]+'_label'),c_class,(c_item.split('.')[0]+'.seg'))
		pts = np.loadtxt(self.data_paths[index], dtype = np.float32)
		pts = torch.from_numpy(pts)
		points_sel = np.random.choice(range(0,len(pts)-1), size=2000, replace=True) # All batches need to have same number of points. This will sample from all the points uniformly and keep them all equal size
		label = np.loadtxt(label_path, dtype = np.long)
		
		pts = pts[points_sel, :]
		label = label[points_sel]-1
		label = torch.from_numpy(label)
		item_class = torch.from_numpy(np.array([self.label[self.classes[c_class]]]))
		return pts, label, item_class

	def __len__(self):
		return len(self.data_paths)

if __name__ == '__main__':
	# classify_net = ShapeNetClassify('ShapeNet', 'train')
	# points, item_class = classify_net.__getitem__(50)
	# class_len = classify_net.__len__()

	# print("current class: {}".format(item_class))
	# print("class length: {}".format(class_len))
	# print("points: {}".format(points))
	# loader = data.DataLoader(classify_net, batch_size = 16, shuffle=True, drop_last=True)
	
	# for batch_num, (sample, label) in enumerate(loader):
	# 	print("{}: {}".format(batch_num, sample))

	segment_net = ShapeNetSegment('ShapeNet', 'train')
	points, labels = segment_net.__getitem__(0)
	class_len = segment_net.__len__()

	for i, k in zip(points, labels):
		print("Point: {}, Label: {}".format(i, k.item()))
















