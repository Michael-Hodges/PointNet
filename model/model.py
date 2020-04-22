import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Vanilla_Classify_Net(nn.Module):
	def __init__(self, output_dim):
		super(Vanilla_Classify_Net, self).__init__()
		# print("Vanilla init not yet implemented")
		# Input Transform
		self.conv1d_1 = nn.Conv1d(3,64,1)
		self.conv1d_2 = nn.Conv1d(64,64,1)

		#Feature Transform and split to segmentation

		self.conv1d_3 = nn.Conv1d(64, 128, 1)
		self.conv1d_4 = nn.Conv1d(128, 1024, 1)
		# Max Pool
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, output_dim)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(128)
		self.bn4 = nn.BatchNorm1d(1024)

		self.bn5 = nn.BatchNorm1d(512)
		self.bn6 = nn.BatchNorm1d(256)
		# Global Features
		# self.fc6 = nn.Linear(PLACEHOLDER, 512, bias=False
	
	def forward(self, x):
		# Define Forward pass
		# print("input {}".format(x.shape)) 
		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# Conv1 size [batch_size, 64, points]
		# print("conv1 {}".format(x.shape))
		x = F.relu(self.bn2(self.conv1d_2(x)))
		# Conv2 size [batch_size, 64,points]
		# print("conv2 {}".format(x.shape))
		x = F.relu(self.bn3(self.conv1d_3(x)))
		# Conv3 size [batch_size, 128,points]
		# print("conv3 {}".format(x.shape))
		x = F.relu(self.bn4(self.conv1d_4(x)))
		# Conv4 size [batch_size, 1024,points]
		# print("conv4 {}".format(x.shape))
		# Perform max pooling

		x, _ = torch.max(x, 2, keepdim=True)
		# max size = [batch_size, 1024, 1]
		# print("max {}".format(x.shape))
		x = x.view(-1, 1024)
		# view size = [batch_size, 1024]
		# print("view {}".format(x.shape))
		x = F.relu(self.bn5(self.fc1(x)))
		x = F.relu(self.bn6(self.fc2(x)))
		x = F.softmax(self.fc3(x), dim=1)
		#output size = [batch_size, output_dim]
		# print("softamx: {}".format(x.shape)) 
		return x

class Classify_Net(nn.Module):
	def __init__(self, output_dim):
		super(Classify_Net, self).__init__()
		print("Classify init not yet implemented")
		pass

	
	def forward(self, x):
		# Define Forward pass input is nx3
		pass


class Vanilla_Segment_Net(nn.Module):
	def __init__(self, output_dim):
		super(Vanilla_Segment_Net, self).__init__()
		# print("Vanilla segment init not yet Finalized")
		# TODO: ALL OF THIS IS COPIED FROM VANILLA_NET. CAN REUSE
		self.conv1d_1 = nn.Conv1d(3,64,1)
		self.conv1d_2 = nn.Conv1d(64,64,1)
		#Feature Transform and split to segmentation
		self.conv1d_3 = nn.Conv1d(64, 128, 1)
		self.conv1d_4 = nn.Conv1d(128, 1024, 1)
		# Max Pool

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(128)
		self.bn4 = nn.BatchNorm1d(1024)
		## END VANILLA_NET
		##START SEGMENTATION NETWORK
		self.seg_conv1 = nn.Conv1d(1088, 512, 1)
		self.seg_conv2 = nn.Conv1d(512, 256, 1)
		self.seg_conv3 = nn.Conv1d(256, 128, 1)
		self.seg_conv4 = nn.Conv1d(128, output_dim, 1)

		self.seg_bn1 = nn.BatchNorm1d(512)
		self.seg_bn2 = nn.BatchNorm1d(256)
		self.seg_bn3 = nn.BatchNorm1d(128)

	def forward(self, x):
		# Define Forward pass
		batch_size = x.shape[0]
		num_points = x.shape[2]
		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# Conv1 size [batch_size, 64, points]	
		input_feats = F.relu(self.bn2(self.conv1d_2(x)))
		# Conv2 size [batch_size, 64,points]
		x = F.relu(self.bn3(self.conv1d_3(input_feats)))
		# Conv3 size [batch_size, 128,points]
		x = F.relu(self.bn4(self.conv1d_4(x)))
		# Conv4 size [batch_size, 1024,points]
		# Perform max pooling
		x, _ = torch.max(x, 2, keepdim=True)
		# max size = [batch_size, 1024, 1]

		glob_feat = x.view(batch_size, 1024, 1)
		glob_feat = glob_feat.repeat(1,1, num_points)
		# glob_feat = [batch_size, 1024, 2000]

		#Begin Segmentation network
		# input feats [n, 64, points] glob feats, [batch_size, 1024, ?]
		segment_input = torch.cat((input_feats, glob_feat), dim = 1)
		# segment input [n, 1088, points]
		x = F.relu(self.seg_bn1(self.seg_conv1(segment_input)))
		# seg_conv1 [n, 512, points]
		x = F.relu(self.seg_bn2(self.seg_conv2(x)))
		# seg_conv2 [n, 256, points]
		x = F.relu(self.seg_bn3(self.seg_conv3(x)))
		# seg_conv3 [n, 128, points]
		x = F.softmax(self.seg_conv4(x), dim = 2)
		
		#seg_conv4/output [nxm]
		return x

class Segment_Net(nn.Module):
	def __init__(self, output_dim):
		super(Segment_Net, self).__init__()
		print("Segment init not yet implemented")

	def forward(self, x):
		print("Segment forward not yet implemented")

if __name__ == '__main__':
	# test_model = Vanilla_Classify_Net(16)
	# test_data = Variable(torch.rand(32,3,2000))
	# output = test_model(test_data)


	test_model2 = Vanilla_Segment_Net(16)
	test_data2 = Variable(torch.rand(32,3,2000))
	output2 = test_model2(test_data2)
	
