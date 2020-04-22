""" Wrapper functions for Pytorch layers.

"""

import numpy as np
import torch


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
    conv2d = torch.nn.Conv2d(inputs.shape[1], num_output_channels, kernel_size, stride, padding).to(torch.device('cuda'))
    kernel_shape = [num_output_channels, inputs.shape[1], kernel_size[0], kernel_size[1]]
    weights = _variable_with_initializer(kernel_shape, stddev, on_gpu=True, use_xavier=use_xavier)
    biases = torch.zeros([num_output_channels])
    conv2d.weight = torch.nn.Parameter(weights)
    conv2d.bias = torch.nn.Parameter(biases)
    output = conv2d(inputs)
    if bn:
        output = batch_norm_2d(output, momentum)
    return output


def max_pool_2d(inputs, kernel_size, stride=[2, 2], padding=0):
    max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return max_pool(inputs)


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

