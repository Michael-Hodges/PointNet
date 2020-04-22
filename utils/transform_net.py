import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
from utils.tf_utils import conv_2d, max_pool_2d, linear


def input_transform_net(point_cloud, momentum=0, k=3):
    # Returns transformation matrix of size 3xK
    batch_size = point_cloud.shape[0]
    num_point = point_cloud.shape[1]

    input_image = point_cloud.unsqueeze(-1)
    net = conv_2d(input_image, 64, [1, 3], 1, padding=1, bn=True, momentum=momentum)
    net = conv_2d(net, 128, [1, 1], 1, padding=1, bn=True, momentum=momentum)
    net = conv_2d(net, 1024, [1, 1], padding=1, bn=True, momentum=momentum)
    net = max_pool_2d(net, [num_point, 1], padding=1)
    net = net.view([batch_size, -1])
    net = linear(net, 512, activation=F.relu, bn=True, momentum=momentum)
    net = linear(net, 256, activation=F.relu, bn=True, momentum=momentum)
    assert(k == 3)
    weights = torch.zeros([256, 3*k], dtype=torch.float32)
    bias = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
    transform = torch.matmul(net, weights)
    transform = torch.add(transform, bias)
    transform = transform.view([batch_size, 3, k])
    return transform






def tfinput_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform
