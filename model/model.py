import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils.transform_net import input_transform_net


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

		self.transform_weights = torch.zeros([256, 3 * k], dtype=torch.float32)
		self.transform_bias = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
		self.transform_weights2 = torch.zeros([256, 64 * 64], dtype=torch.float32)
		self.transform_bias2 = torch.tensor([64*64], dtype=torch.float32)


	def forward(self, x, momentum=0.95, k=3, DEVICE=torch.device('cuda')):
		# Define Forward pass
		# print("input {}".format(x.shape))
		batch_size = x.shape[0]
		num_point = x.shape[1]

		input_image = x.view(batch_size, 3, num_point)
		# net = conv_2d(input_image, 64, [1, 3], 1, padding=1, bn=True, momentum=momentum)
		net = F.relu(self.transform_batch_norm_2d1(self.transform_conv1(input_image)))
		net = F.relu(self.transform_batch_norm_2d2(self.transform_conv2(net)))
		net = F.relu(self.transform_batch_norm_2d3(self.transform_conv3(net)))
		net = torch.max(net, 2, keepdim=True)[0]
		# net = max_pool_2d(net, [num_point, 1], padding=1)
		net = net.view(-1, 1024)
		net = F.relu(self.transform_batch_norm_2d5(self.fc1(net)))
		net = F.relu(self.transform_batch_norm_2d4(self.fc2(net)))
		assert (k == 3)
		transform = torch.matmul(net, self.transform_weights.to(DEVICE))
		transform = torch.add(transform, self.transform_bias.to(DEVICE))
		transform = transform.view([batch_size, 3, k])
		x = torch.matmul(x, transform)
		x = x.view(batch_size, 3, -1)
		# input [batch_size, input_dim, points]
		x = F.relu(self.bn1(self.conv1d_1(x))) #nx64
		# Conv1 size [batch_size, 64, points]
		# print("conv1 {}".format(x.shape))
		x = F.relu(self.bn2(self.conv1d_2(x)))
		# Conv2 size [batch_size, 64,points]
		input = x.view(batch_size, -1,  num_point)
		net = F.relu(self.transform_batch_norm_2d1(self.transform_conv4(input)))
		net = F.relu(self.transform_batch_norm_2d2(self.transform_conv2(net)))
		net = F.relu(self.transform_batch_norm_2d3(self.transform_conv3(net)))
		net = torch.max(net, 2, keepdim=True)[0]
		# net = max_pool_2d(net, [num_point, 1], padding=1)
		net = net.view(-1, 1024)
		net = F.relu(self.transform_batch_norm_2d5(self.fc1(net)))
		net = F.relu(self.transform_batch_norm_2d4(self.fc2(net)))
		transform = torch.matmul(net, self.transform_weights2.to(DEVICE))
		transform = torch.add(transform, self.transform_bias2.to(DEVICE))
		transform = transform.view([batch_size, 64, 64])
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
		x = F.softmax(self.seg_conv4(x), dim = 1)
		
		#seg_conv4/output [nxm]
		return x


class Segment_Net(nn.Module):
	def __init__(self, output_dim):
		super(Segment_Net, self).__init__()
		print("Segment init not yet implemented")

	def forward(self, x):
		print("Segment forward not yet implemented")


def _variable_with_initializer(shape, stddev, use_xavier=False, on_gpu=False, device_id=0):
    var = torch.empty(shape)
    if use_xavier is True:
        initializer = torch.nn.init.xavier_uniform_(var)
    else:
        initializer = torch.nn.init.normal_(var, std=stddev)
    if on_gpu is False:
        var.to(torch.device('cpu'))
    else:
        var.to(torch.device('cuda:' + str(device_id)))
    return var


def batch_norm_2d(inputs, momentum, affine=True, eps=1e-5, track_running_stats=True):
    # Yet to implement the exponential moving average part
    # Have to keep in mind to implement it
    input_shape = inputs.shape[-1]
    batch_norm = torch.nn.BatchNorm2d(input_shape, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)
    return batch_norm(inputs)


def conv_2d(inputs, num_output_channels, kernel_size, stride, padding, stddev=1e-3, bn=False, momentum=0.95,
            use_xavier=True):
    conv2d = torch.nn.Conv2d(inputs, num_output_channels, kernel_size, stride, padding)
    kernel_shape = [num_output_channels, inputs, kernel_size[0], kernel_size[1]]
    weights = _variable_with_initializer(kernel_shape, stddev, use_xavier=use_xavier)
    biases = torch.zeros([num_output_channels])
    conv2d.weight = torch.nn.Parameter(weights)
    conv2d.bias = torch.nn.Parameter(biases)
    # output = conv2d(inputs)
    # if bn:
    #     output = batch_norm_2d(output, momentum)
    return conv2d


# def conv_1d(inputs, num_output_channels, kernel_size, stride, padding, stddev=1e-3, bn=False, momentum=0.95,
#             use_xavier=True):
#     conv1d = torch.nn.Conv1d(inputs, num_output_channels, kernel_size, stride, padding)
#     # kernel_shape = [num_output_channels, inputs, kernel_size[0], kernel_size[1]]
#     # weights = _variable_with_initializer(kernel_shape, stddev, use_xavier=use_xavier)
#     biases = torch.zeros([num_output_channels])
#     # conv1d.weight = torch.nn.Parameter(weights)
#     conv1d.bias = torch.nn.Parameter(biases)
#     # output = conv2d(inputs)
#     # if bn:
#     #     output = batch_norm_2d(output, momentum)
#     return conv1d


# def max_pool_2d(inputs, kernel_size, stride=[2, 2], padding=0):
#     max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
#     return max_pool(inputs)


def batch_norm_fc(inputs, momentum, affine=True, eps=1e-5, track_running_stats=True):
    input_shape = inputs.shape[-1]
    batch_norm = torch.nn.BatchNorm1d(input_shape)
    return batch_norm(inputs)


def linear(inputs, num_outputs, activation, use_xavier=True, bn=False, momentum=0.95):
    weights = _variable_with_initializer([inputs.shape[-1], num_outputs], stddev, use_xavier=use_xavier)
    outputs = torch.matmul(inputs, weights)
    bias = torch.zeros([num_outputs])
    outputs = torch.add(outputs, bias)
    if bn:
        outputs = batch_norm_fc(outputs, momentum=momentum)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


if __name__ == '__main__':
	# test_model = Vanilla_Classify_Net(16)
	# test_data = Variable(torch.rand(32,3,2000))
	# output = test_model(test_data)


	test_model2 = Vanilla_Segment_Net(16)
	test_data2 = Variable(torch.rand(32,3,2000))
	output2 = test_model2(test_data2)
	
