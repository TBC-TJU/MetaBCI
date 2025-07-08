# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/7/06
# License: MIT License
"""
ShallowFBCSP.
Modified from https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py

"""

from collections import OrderedDict

import torch
import torch.nn as nn
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


@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class ShallowNet(nn.Module):

    """
    ShallowNet is a neural network structure specifically designed for motion imagination (MI) tasks,
    decoding the band power features in MI signals. [1]_

    ShallowNet uses two convolution layers to simulate bandpass filtering and spatial filtering in the FBCSP(Filter
    Bank Common Spatial Pattern) algorithm.
    The first layer in ShallowNet performs convolution on the time dimension,
    convolving the EEG data in each channel separately to extract time domain features.
    The second layer integrates the features of each channel extracted by the first layer through convolution
    across channels. ShallowNet also designed an average pooling layer after the two convolution layers,
    and two activation functions :math:`x^2` and  :math:`log(x)` respectively is applied before and after
    the average pool layer,
    referring to experimental log-variance calculations in the FBCSP algorithm.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-07-06

    update log:
        2023-12-11 by MutexD <wudf@tju.edu.cn>

    Parameters
    ----------
    n_channels: int
        Lead count for the input signal.
    n_samples: int
        Sampling points of the input signal. The value equals sampling rate (Hz) * signal duration (s).
    n_classes: int
        The number of classes of input signals to be classified.

    Attributes
    ----------
    step1: torch.nn.Sequential
        First convolution layer
    step2: torch.nn.Sequential
        Second convolution layer
    step3: torch.nn.Sequential
        Pooling Layer and Flattening operation
    fc_layer: torch.nn.Linear
        linear connection layer for classification.
    model: torch.nn.Sequential
        stacked model layers


    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> num_classes = 2
    >>> estimator = ShallowNet(X.shape[1], X.shape[2], num_classes)
    >>> estimator.fit(X[train_index], y[train_index])

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    References
    ----------
    .. [1] Schirrmeiste R T , Springenberg J T , Fiedere L , et al. Deep learning with convolutional neural networks
       for EEG decoding and visualization[J]. Human Brain Mapping, 2017.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        # super(ShallowNet, self).__init__()
        super().__init__()

        n_time_filters = 40
        time_kernel = 25
        n_space_filters = 40
        pool_kernel = 75
        pool_stride = 15
        dropout_rate = 0.5

        # temporal convolution
        self.step1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "time_conv",
                        nn.Conv2d(
                            1,
                            n_time_filters,
                            (1, time_kernel),
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    )
                ]
            )
        )

        # spatial convolution
        self.step2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "space_conv",
                        nn.Conv2d(
                            n_time_filters,
                            n_space_filters,
                            (n_channels, 1),
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(n_space_filters)),
                ]
            )
        )

        # pooling
        self.step3 = nn.Sequential(
            OrderedDict(
                [
                    ("square", Square()),
                    (
                        "avg_pool",
                        nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
                    ),
                    ("safe_log", SafeLog()),
                    ("drop", nn.Dropout(p=dropout_rate)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, n_channels, n_samples))
            fake_output = self.step3(self.step2(self.step1(fake_input)))
            middle_size = fake_output.shape[-1]

        self.fc_layer = nn.Linear(middle_size, n_classes, bias=True)
        self.model = nn.Sequential(self.step1, self.step2, self.step3, self.fc_layer)
        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
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

    def cal_backbone(self, X: Tensor, **kwargs):
        X = X.unsqueeze(1)
        tmp = X
        for i in range(len(self.model) - 1):
            tmp = self.model[i](tmp)
        return tmp
