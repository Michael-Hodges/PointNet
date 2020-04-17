import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Vanilla_Classify_Net(nn.Module):
	def __init__(self, output_dim):
		super(Vanilla_Classify_Net, self).__init__()
		print("Vanilla init not yet implemented")
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
		# Define Forward pass input is nx3
		print("input {}".format(x.shape)) 
		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# size [batch_size, 64, points]
		print("conv1 {}".format(x.shape))
		x = F.relu(self.bn2(self.conv1d_2(x)))
		print("conv2 {}".format(x.shape))
		x = F.relu(self.bn3(self.conv1d_3(x)))
		print("conv3 {}".format(x.shape))
		x = F.relu(self.bn4(self.conv1d_4(x)))
		print("conv4 {}".format(x.shape))
		# Perform max pooling

		x, _ = torch.max(x, 2, keepdim=True)
		print("max {}".format(x.shape))
		x = x.view(-1, 1024)
		print("view {}".format(x.shape))
		x = F.relu(self.bn5(self.fc1(x)))
		x = F.relu(self.bn6(self.fc2(x)))
		x = F.softmax(self.fc3(x), dim=1)
		print("softamx: {}".format(x.shape)) #[batch_size, output_dim]
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
		print("Vanilla segment init not yet implemented")

	def forward(self, x):
		Print("Vanilla segment forward not yet implemented")

class Segment_Net(nn.Module):
	def __init__(self, output_dim):
		super(Segment_Net, self).__init__()
		print("Segment init not yet implemented")

	def forward(self, x):
		Print("Segment forward not yet implemented")

if __name__ == '__main__':
	test_model = Vanilla_Classify_Net(16)
	test_data = Variable(torch.rand(32,3,2000))
	output = test_model(test_data)
	
