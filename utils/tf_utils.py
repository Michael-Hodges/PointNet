""" Wrapper functions for Pytorch layers.

"""

import numpy as np
import torch


def _variable_with_initializer(shape, stddev, use_xavier=False, on_gpu=False, device_id=0):
    var = torch.Tensor(shape)
    if use_xavier is True:
        initializer = torch.nn.init.xavier_uniform_(var)
    else:
        initializer = torch.nn.init.normal_(var)
    if on_gpu is False:
        var.to(torch.device('cpu'))
    else:
        var.to(torch.device('cuda:' + str(device_id)))
    return var



