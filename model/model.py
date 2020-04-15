import torch
import torch.nn as nn
import torch.nn.functional as functional


class Vanilla_Classify_Net(nn.module):
	def __init__(self, output_dim):
		super(Vanilla_Classify_Net, self).__init__()
		print("Vanilla init not yet implemented")
		# Input Transform
		self.fc1 = nn.linear(3, 64, bias=False)
		self.fc2 = nn.linear(64, 64, bias=False)
		#Feature Transform and split to segmentation
		self.fc3 = nn.linear(64, 64, bias=False)
		self.fc4 = nn.linear(64, 128, bias=False)
		self.fc5 = nn.linear(128, 1024, bias=False)
		self.mp1 = nn.maxpool()
		# Global Features
		self.fc6 = nn.linear(PLACEHOLDER, 512, bias=False)
		self.fc7 = nn.linear(512, 256, bias=False)
		self.fc8 = nn.linear(256, output_dim, bias=False)
	
	def forward(self, x):
		# Define Forward pass input is nx3
		x = F.relu(self.fc1(x)) #nx64
		x = F.relu(self.fc2(x)) #nx64
		x = F.relu(self.fc3(x))	#nx64
		x = F.relu(self.fc4(x)) #nx128
		x = F.relu(self.fc5(x)) #nx1024
		x = self.mp1(x)			#1x1024
		x = F.relu(self.fc6(x)) #1x512
		x = F.relu(self.fc7(x)) #1x256
		x = F.relu(self.fc8(x)) #1xoutput_dim

class Classify_Net(nn.module):
	def __init__(self, output_dim):
		super(Classify_Net, self).__init__()
		print("Classify init not yet implemented")
		# Input Transform
		self.fc1 = nn.linear(3, 64, bias=False)
		self.fc2 = nn.linear(64, 64, bias=False)
		#Feature Transform and split to segmentation
		self.fc3 = nn.linear(64, 64, bias=False)
		self.fc4 = nn.linear(64, 128, bias=False)
		self.fc5 = nn.linear(128, 1024, bias=False)
		self.mp1 = nn.maxpool()
		# Global Features
		self.fc6 = nn.linear(PLACEHOLDER, 512, bias=False)
		self.fc7 = nn.linear(512, 256, bias=False)
		self.fc8 = nn.linear(256, output_dim, bias=False)
	
	def forward(self, x):
		# Define Forward pass input is nx3
		x = F.relu(self.fc1(x)) #nx64
		x = F.relu(self.fc2(x)) #nx64
		x = F.relu(self.fc3(x))	#nx64
		x = F.relu(self.fc4(x)) #nx128
		x = F.relu(self.fc5(x)) #nx1024
		x = self.mp1(x)			#1x1024
		x = F.relu(self.fc6(x)) #1x512
		x = F.relu(self.fc7(x)) #1x256
		x = F.relu(self.fc8(x)) #1xoutput_dim

class Vanilla_Segment_Net(nn.module):
	def __init__(self, output_dim):
		super(Vanilla_Segment_Net, self).__init__()
		print("Vanilla segment init not yet implemented")

	def forward(self, x):
		Print("Vanilla segment forward not yet implemented")

class Segment_Net(nn.module):
	def __init__(self, output_dim):
		super(Segment_Net, self).__init__()
		print("Segment init not yet implemented")

	def forward(self, x):
		Print("Segment forward not yet implemented")

if __name__ == '__main__':
	test_model = Vanilla_Classify_Net(10)
	print(test_model)
	pass
