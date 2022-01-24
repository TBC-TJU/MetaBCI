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
import torch.nn.functional as F

from .base import _glorot_weight_zero_bias, compute_same_pad2d, SkorchNet

class _CorrLayer(nn.Module):
    def __init__(self):
        super(_CorrLayer, self).__init__()
    
    def forward(self, X, T):
        # X: n_batch, 1, 1, n_samples
        # T: n_batch, 1, n_classes, n_samples
        T = torch.swapaxes(T, -1, -2)
        corr_xt = torch.matmul(X, T) # n_batch, 1, 1, n_classes
        corr_xx = torch.sum(torch.square(X), -1, keepdim=True)
        corr_tt = torch.sum(torch.square(T), -2, keepdim=True)
        corr = corr_xt / (torch.sqrt(corr_xx) * torch.sqrt(corr_tt))
        return corr


@SkorchNet
class ConvCA(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, 
            n_time_filters_signal=16,
            n_time_filters_template=40, 
            time_conv_kernel=9,
            dropout_signal=0.75, dropout_template=0.15):
        # super(ConvCA, self).__init__()
        super().__init__()

        # input: n_batch, 1, n_channel, n_samples
        self.signal_cnn = nn.Sequential(OrderedDict([
            ('same_padding1',
            nn.ConstantPad2d(
                compute_same_pad2d(
                    (n_channels, n_samples), 
                    (n_channels, time_conv_kernel), 
                    stride=(1, 1)), 
                0)),
            ('conv1',
            nn.Conv2d(1, n_time_filters_signal, (n_channels, time_conv_kernel),
                stride=(1, 1), padding=0, bias=True)),
            ('same_padding2',
            nn.ConstantPad2d(
                compute_same_pad2d(
                    (n_channels, n_samples), 
                    (n_channels, 1), 
                    stride=(1, 1)), 
                0)),
            ('conv2',
            nn.Conv2d(n_time_filters_signal, 1, (n_channels, 1),
                stride=(1, 1), padding=0, bias=True)),
            ('conv3',
            nn.Conv2d(1, 1, (n_channels, 1),
                stride=(1, 1), padding=0, bias=True)),
            ('dropout', nn.Dropout(dropout_signal)),
        ]))

        # input: n_batch, n_channels, n_classes, n_samples
        self.template_cnn = nn.Sequential(OrderedDict([
            ('same_padding1',
            nn.ConstantPad2d(
                compute_same_pad2d(
                    (n_classes, n_samples), 
                    (1, time_conv_kernel), 
                    stride=(1, 1)), 
                0)),
            ('conv1',
            nn.Conv2d(n_channels, n_time_filters_template, (1, time_conv_kernel),
                stride=(1, 1), padding=0, bias=True)),
            ('same_padding2',
            nn.ConstantPad2d(
                compute_same_pad2d(
                    (n_classes, n_samples), 
                    (1, time_conv_kernel), 
                    stride=(1, 1)), 
                0)),
            ('conv2',
            nn.Conv2d(n_time_filters_template, 1, (1, time_conv_kernel),
                stride=(1, 1), padding=0, bias=True)),
            ('dropout', nn.Dropout(dropout_template))
        ]))

        self.corr_layer = _CorrLayer()
        self.flatten_layer = nn.Flatten()
        self.fc_layer = nn.Linear(n_classes, n_classes)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
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