import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils.transform_net import input_transform_net


class Transform(nn.Module):
	def __init__(self, type="input", k=3):
		super(Transform, self).__init__()
		self.type = type
		self.K = k
		self.transform_conv1 = nn.Conv1d(3, 64, 1, 1, padding=1)
		self.transform_conv2 = nn.Conv1d(64, 128, 1, 1, padding=1)
		self.transform_conv3 = nn.Conv1d(128, 1024, 1, 1, padding=1)
		self.transform_conv4 = nn.Conv1d(64, 64, 1, 1, padding=1)
		self.transform_max_pool = nn.MaxPool2d([1024, 1], [2, 2], padding=0)
		self.transform_batch_norm_2d1 = nn.BatchNorm1d(64, momentum=0.95)
		self.transform_batch_norm_2d2 = nn.BatchNorm1d(128, momentum=0.95)
		self.transform_batch_norm_2d3 = nn.BatchNorm1d(1024, momentum=0.95)
		self.transform_batch_norm_2d4 = nn.BatchNorm1d(256, momentum=0.95)
		self.transform_batch_norm_2d5 = nn.BatchNorm1d(512, momentum=0.95)
		self.transform_weights = nn.Parameter(torch.zeros([256, 3 * k], dtype=torch.float32))
		self.transform_bias = nn.Parameter(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))
		self.transform_weights2 = nn.Parameter(torch.zeros([256, k * k], dtype=torch.float32))
		self.transform_bias2 = nn.Parameter(torch.tensor([k * k], dtype=torch.float32))
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)

	def forward(self, point_cloud, batch_size, num_point):
		input_image = point_cloud.view(batch_size, self.K, num_point)
		# net = conv_2d(input_image, 64, [1, 3], 1, padding=1, bn=True, momentum=momentum)
		if self.type == "input":
			net = F.relu(self.transform_batch_norm_2d1(self.transform_conv1(input_image)))
		if self.type == "feature":
			net = F.relu(self.transform_batch_norm_2d1(self.transform_conv4(input_image)))
		net = F.relu(self.transform_batch_norm_2d2(self.transform_conv2(net)))
		net = F.relu(self.transform_batch_norm_2d3(self.transform_conv3(net)))
		net = torch.max(net, 2, keepdim=True)[0]
		# net = max_pool_2d(net, [num_point, 1], padding=1)
		net = net.view(-1, 1024)
		net = F.relu(self.transform_batch_norm_2d5(self.fc1(net)))
		net = F.relu(self.transform_batch_norm_2d4(self.fc2(net)))
		if self.type == "input":
			assert (self.K == 3)
			transform = torch.matmul(net, self.transform_weights)
			transform = torch.add(transform, self.transform_bias)
			transform = transform.view([batch_size, 3, self.K])
		elif self.type == "feature":
			transform = torch.matmul(net, self.transform_weights2)
			transform = torch.add(transform, self.transform_bias2)
			transform = transform.view([batch_size, self.K, self.K])
		else:
			transform = torch.matmul(net, self.transform_weights)
			transform = torch.add(transform, self.transform_bias)
			transform = transform.view([batch_size, 3, self.K])
		return transform


class Vanilla_Classify_Net(nn.Module):
	def __init__(self, output_dim, k=3):
		super(Vanilla_Classify_Net, self).__init__()
		# print("Vanilla init not yet implemented")
		# Input Transform)
		self.conv1d_1 = nn.Conv1d(3, 64, 1)
		self.conv1d_2 = nn.Conv1d(64, 64, 1)

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

		# Transform Networks
		self.input_transform = Transform(type="input")
		self.feature_transform = Transform(type="feature", k=64)


	def forward(self, x, momentum=0.95, k=3, DEVICE=torch.device('cuda')):
		# Define Forward pass
		# print("input {}".format(x.shape))
		batch_size = x.shape[0]
		num_point = x.shape[1]

		# Input Transform
		transform = self.input_transform(x, batch_size, num_point)
		x = torch.matmul(x, transform)
		x = x.view(batch_size, 3, -1)

		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# Conv1 size [batch_size, 64, points]
		# print("conv1 {}".format(x.shape))
		x = F.relu(self.bn2(self.conv1d_2(x)))
		# Conv2 size [batch_size, 64,points]

		# Feature Transform
		transform = self.feature_transform(x, batch_size, num_point)
		x = torch.matmul(x.view(batch_size, num_point, -1), transform)
		x = x.view(batch_size, -1, num_point)

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
		self.seg_conv4 = nn.Conv1d(128, 50, 1)
		self.seg_conv5 = nn.Conv1d(50, output_dim, 1)


		self.seg_bn1 = nn.BatchNorm1d(512)
		self.seg_bn2 = nn.BatchNorm1d(256)
		self.seg_bn3 = nn.BatchNorm1d(128)
		self.seg_bn4 = nn.BatchNorm1d(50)

		# Transform Networks
		self.input_transform = Transform(type="input")
		self.feature_transform = Transform(type="feature", k=64)

	def forward(self, x):
		# Define Forward pass
		batch_size = x.shape[0]
		num_point = x.shape[1]

		# Input Transform
		transform = self.input_transform(x, batch_size, num_point)
		x = torch.matmul(x, transform)
		x = x.view(batch_size, 3, -1)

		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# Conv1 size [batch_size, 64, points]
		x = F.relu(self.bn2(self.conv1d_2(x))) #64x64
		# Conv2 size [batch_size, 64, points]

		# Feature Transform
		transform = self.feature_transform(x, batch_size, num_point)
		x = torch.matmul(x.view(batch_size, num_point, -1), transform)
		input_feats = x.view(batch_size, -1, num_point)

		x = F.relu(self.bn2(self.conv1d_2(input_feats)))
		# Conv2 size [batch_size, 64, points]
		x = F.relu(self.bn3(self.conv1d_3(x)))
		# Conv3 size [batch_size, 128,points]
		x = F.relu(self.bn4(self.conv1d_4(x)))
		# Conv4 size [batch_size, 1024,points]
		# Perform max pooling
		x, _ = torch.max(x, 2, keepdim=True)
		# max size = [batch_size, 1024, 1]

		glob_feat = x.view(batch_size, 1024, 1)
		glob_feat = glob_feat.repeat(1,1, num_point)
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
		x = F.relu(self.seg_bn4(self.seg_conv4(x)))
		# seg_conv4 [n, 50, points]
		x = F.softmax(self.seg_conv5(x), dim = 1)
		# seg_conv5/output [nxm]
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
	
