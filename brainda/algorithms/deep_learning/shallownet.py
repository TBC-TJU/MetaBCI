# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/7/06
# License: MIT License
"""
ShallowFBCSP.
Modified from https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py

"""

from typing import Optional, Dict, List, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base import SkorchNet

class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()
    
    def forward(self, X):
        return torch.square(X)


class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super(SafeLog, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        return torch.log(torch.clamp(X, min=self.eps))


@SkorchNet
class ShallowNet(nn.Module):
    def __init__(self,
            n_channels: int,
            n_samples: int,
            n_classes: int,
            n_time_filters=40,
            time_kernel=25,
            n_space_filters=40,
            pool_kernel=75,
            pool_stride=15,
            dropout_rate=0.5):
        # super(ShallowNet, self).__init__()
        super().__init__()

        # temporal convolution
        self.step1 = nn.Sequential(OrderedDict([
            ('time_conv', 
            nn.Conv2d(
                1, n_time_filters, (1, time_kernel),
                stride=1,
                padding=0,
                bias=True))
        ]))
        
        # spatial convolution
        self.step2 = nn.Sequential(OrderedDict([
            ('space_conv',
            nn.Conv2d(
                n_time_filters, n_space_filters, (n_channels, 1),
                stride=1,
                padding=0,
                bias=False)),
            ('bn', nn.BatchNorm2d(n_space_filters))
        ]))

        # pooling
        self.step3 = nn.Sequential(OrderedDict([
            ('square',Square()),
            ('avg_pool',
            nn.AvgPool2d(
                (1, pool_kernel),
                stride=(1, pool_stride))),
            ('safe_log', SafeLog()),
            ('drop', nn.Dropout(p=dropout_rate)),
            ('flatten', nn.Flatten())
        ]))

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, n_channels, n_samples))
            fake_output = self.step3(self.step2(self.step1(fake_input)))
            middle_size = fake_output.shape[-1]

        self.fc_layer = nn.Linear(middle_size, n_classes, bias=True)
        self.model = nn.Sequential(
            self.step1, 
            self.step2, 
            self.step3, 
            self.fc_layer)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.step1.time_conv.weight, gain=1)
        nn.init.constant_(self.step1.time_conv.bias, 0)
        nn.init.xavier_uniform_(self.step2.space_conv.weight, gain=1)
        nn.init.constant_(self.step2.bn.weight, 1)
        nn.init.constant_(self.step2.bn.bias, 0)
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1)
        nn.init.constant_(self.fc_layer.bias, 0)

    def forward(self, X: Tensor, **kwargs):
        X = X.unsqueeze(1)
        out = self.model(X)
        return out
