# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/9/12
# License: MIT License
"""
Conv-CA
Modified from https://github.com/yaoli90/Conv-CA

"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import _glorot_weight_zero_bias, compute_same_pad2d, SkorchNet


class _CorrLayer(nn.Module):
    def __init__(self):
        super(_CorrLayer, self).__init__()

    def forward(self, X, T):
        # X: n_batch, 1, 1, n_samples
        # T: n_batch, 1, n_classes, n_samples
        T = torch.swapaxes(T, -1, -2)
        corr_xt = torch.matmul(X, T)  # n_batch, 1, 1, n_classes
        corr_xx = torch.sum(torch.square(X), -1, keepdim=True)
        corr_tt = torch.sum(torch.square(T), -2, keepdim=True)
        corr = corr_xt / (torch.sqrt(corr_xx) * torch.sqrt(corr_tt))
        return corr


@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class ConvCA(nn.Module):
    """
    ConvCA is a neural network designed for SSVEP task based on TRCA algorithm.
    It uses three convolutional layers to extract input signal features. [1]_
    And two convolutional layers were used to extract the features of reference signals (the average value
    of all training data for each class),
    the correlation coefficients of the features of these two types of signals were calculated as decision values,
    and the linear layer was used for classification.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-9-12

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
    signal_cnn: torch.nn.Sequential
        A CNN block for processing input signals
    template_cnn: torch.nn.Sequential
        A CNN block for processing template signals
    corr_layer: torch.nn.Module
        Correlation Calculate layer
    flatten_layer: torch.nn.Linear
        Dense connection layer for classification.
    fc_layer: torch.nn.Module
        Reshape input tensor from 3D to 2D

    Examples
    ----------
    >>> # for convCA, you will also need a T(reference signal), you can initialize network like
    >>> # shallownet by estimator = ConvCA(X.shape[1], X.shape[2], 2),
    >>> # but you need to wrap X and T in a dict like this {'X': X, 'T', T} to train the network
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> # T size: [batch size, number of channels, number of classes, number of sample points]
    >>> num_classes = 2
    >>> num_sub_bands = 3
    >>> estimator = ConvCA(X.shape[1], X.shape[2], 2)
    >>> dict_ = {'X': train_X, 'T': T}
    >>> estimator.fit(dict_, train_Y)

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    References
    ----------
    .. [1] Li Y , Xiang J , Kesavadas T . Convolutional Correlation Analysis for Enhancing the Performance of
       SSVEP-Based Brain-Computer Interface[J].
       IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020.

    """
    def __init__(self, n_channels, n_samples, n_classes):
        # super(ConvCA, self).__init__()
        super().__init__()

        n_time_filters_signal = 16
        n_time_filters_template = 40
        time_conv_kernel = 9
        dropout_signal = 0.75
        dropout_template = 0.15

        # input: n_batch, 1, n_channel, n_samples
        self.signal_cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding1",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_channels, n_samples),
                                (n_channels, time_conv_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "conv1",
                        nn.Conv2d(
                            1,
                            n_time_filters_signal,
                            (n_channels, time_conv_kernel),
                            stride=(1, 1),
                            padding=0,
                            bias=True,
                        ),
                    ),
                    (
                        "same_padding2",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_channels, n_samples), (n_channels, 1), stride=(1, 1)
                            ),
                            0,
                        ),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            n_time_filters_signal,
                            1,
                            (n_channels, 1),
                            stride=(1, 1),
                            padding=0,
                            bias=True,
                        ),
                    ),
                    (
                        "conv3",
                        nn.Conv2d(
                            1, 1, (n_channels, 1), stride=(1, 1), padding=0, bias=True
                        ),
                    ),
                    ("dropout", nn.Dropout(dropout_signal)),
                ]
            )
        )

        # input: n_batch, n_channels, n_classes, n_samples
        self.template_cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding1",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_classes, n_samples),
                                (1, time_conv_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "conv1",
                        nn.Conv2d(
                            n_channels,
                            n_time_filters_template,
                            (1, time_conv_kernel),
                            stride=(1, 1),
                            padding=0,
                            bias=True,
                        ),
                    ),
                    (
                        "same_padding2",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_classes, n_samples),
                                (1, time_conv_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            n_time_filters_template,
                            1,
                            (1, time_conv_kernel),
                            stride=(1, 1),
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("dropout", nn.Dropout(dropout_template)),
                ]
            )
        )

        self.corr_layer = _CorrLayer()
        self.flatten_layer = nn.Flatten()
        self.fc_layer = nn.Linear(n_classes, n_classes)
        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X, T):
        # X: (n_batch, n_channels, n_samples)
        # T: (n_batch, n_channels, n_classes, n_samples)
        X = X.unsqueeze(1)
        X = self.signal_cnn(X)
        T = self.template_cnn(T)
        corr = self.corr_layer(X, T)
        corr = self.flatten_layer(corr)
        out = self.fc_layer(corr)
        return out
