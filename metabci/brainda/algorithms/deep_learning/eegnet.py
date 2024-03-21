# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
"""
EEGNet.
Modified from https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from .base import (
    compute_same_pad2d,
    MaxNormConstraintLinear,
    MaxNormConstraintConv2d,
    _glorot_weight_zero_bias,
    SkorchNet,
)


class SeparableConv2d(nn.Module):
    """An equally SeparableConv2d in Keras.
    A depthwise conv followed by a pointwise conv.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        D=1,
    ):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels * D,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels * D, out_channels, 1, stride=1, padding=0, bias=bias
        )
        self.model = nn.Sequential(self.depthwise_conv, self.pointwise_conv)

    def forward(self, X):
        return self.model(X)


@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class EEGNet(nn.Module):
    """

    EEGNet is a general EEG deep learning model which performs well in multiple BCI paradigms.
    The EEGNet architecture includes batch regularization, dropout, and ELU structures.
    Several different types of convolutional layers are cleverly designed in EEGNet,
    such as Deep-wise Convolution and Separable Convolution. By applying these convolution layers,
    you can effectively reduce the number of parameters to be fitted and speed up training. [1]_

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-23

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

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> num_classes = 2
    >>> estimator = EEGNet(X.shape[1], X.shape[2], num_classes)
    >>> estimator.fit(X[train_index], y[train_index])

    References
    ----------
    .. [1] Lawhern V J , Solon A J , Waytowich N R , et al. EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces[J]. Journal of Neural Engineering, 2018, 15(5):056013.1-056013.17.

    """

    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()

        time_kernel = (8, (1, 64), (1, 1))
        D = 2
        pool_kernel1 = ((1, 4), (1, 4))
        separa_kernel = (16, (1, 16), (1, 1))
        pool_kernel2 = ((1, 8), (1, 8))
        dropout_rate = 0.5
        fc_norm_rate = 0.25
        depthwise_norm_rate = 1

        bn_affine = True
        # time convolution
        self.step1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_channels, n_samples),
                                time_kernel[1],
                                stride=time_kernel[2],
                            ),
                            0,
                        ),
                    ),
                    (
                        "time_conv",
                        nn.Conv2d(
                            1,
                            time_kernel[0],
                            time_kernel[1],
                            stride=time_kernel[2],
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(time_kernel[0], affine=bn_affine)),
                    ("drop", nn.Dropout(dropout_rate)),
                ]
            )
        )

        # depthwise convolution
        self.step2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "depthwise_conv",
                        MaxNormConstraintConv2d(
                            time_kernel[0],
                            time_kernel[0] * D,
                            (n_channels, 1),
                            groups=time_kernel[0],
                            bias=False,
                            max_norm_value=depthwise_norm_rate,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(time_kernel[0] * D, affine=bn_affine)),
                    ("elu", nn.ELU()),
                    ("ave_pool", nn.AvgPool2d(pool_kernel1[0], stride=pool_kernel1[1])),
                    ("drop", nn.Dropout(dropout_rate)),
                ]
            )
        )

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, n_channels, n_samples))
            fake_output = self.step2(self.step1(fake_input))
            middle_size = fake_output.shape[2:]

        # separable convolution
        self.step3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                middle_size, separa_kernel[1], stride=separa_kernel[2]
                            ),
                            0,
                        ),
                    ),
                    (
                        "separable_conv",
                        SeparableConv2d(
                            time_kernel[0] * D,
                            separa_kernel[0],
                            separa_kernel[1],
                            stride=separa_kernel[2],
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(separa_kernel[0], affine=bn_affine)),
                    ("elu", nn.ELU()),
                    ("ave_pool", nn.AvgPool2d(pool_kernel2[0], pool_kernel2[1])),
                    ("drop", nn.Dropout(dropout_rate)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        with torch.no_grad():
            fake_output = self.step3(fake_output)
            middle_size = fake_output.shape[1]

        self.fc_layer = MaxNormConstraintLinear(
            middle_size, n_classes, max_norm_value=fc_norm_rate
        )

        self.model = nn.Sequential(self.step1, self.step2, self.step3, self.fc_layer)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X):
        X = X.unsqueeze(1)  # 4D
        out = self.model(X)
        return out

    def cal_backbone(self, X: Tensor, **kwargs):
        X = X.unsqueeze(1)
        tmp = X
        for i in range(len(self.model) - 1):
            tmp = self.model[i](tmp)
        return tmp
