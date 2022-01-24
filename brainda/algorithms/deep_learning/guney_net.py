# -*- coding: utf-8 -*-
"""
Guney's network proposed in A Deep Neural Network for SSVEP-based Brain-Computer Interfaces.

Modified from https://github.com/osmanberke/Deep-SSVEP-BCI.git
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import compute_same_pad2d, _narrow_normal_weight_zero_bias, compute_out_size, SkorchNet


@SkorchNet
class GuneyNet(nn.Module):
    """
    Guney's network for decoding SSVEP.
    They used two stages to train the network. 
    
    The first stage is with all training data in the dataset. 
    lr: 1e-4, batch_size: 100, l2_regularization: 1e-3, epochs: 1000
    
    The second stage is a fine-tuning process with each subject's training data.
    lr: 1e-4, batch_size: full size, l2_regularization: 1e-3, epochs: 1000
    spatial_dropout=time1_dropout=0.6
    """
    def __init__(self, n_channels, n_samples, n_classes, n_bands,
        n_spatial_filters=120, spatial_dropout=0.1,
        time1_kernel=2, time1_stride=2, n_time1_filters=120,
        time1_dropout=0.1,
        time2_kernel=10, n_time2_filters=120,
        time2_dropout=0.95):
        # super(GuneyNet, self).__init__()
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_bands = n_bands

        self.model = nn.Sequential(OrderedDict([
            ('band_layer', nn.Conv2d(n_bands, 1, (1, 1), bias=False)),
            ('spatial_layer', nn.Conv2d(1, n_spatial_filters, (n_channels, 1))),
            ('spatial_dropout', nn.Dropout(spatial_dropout)),
            ('time1_layer', 
                nn.Conv2d(n_spatial_filters, n_time1_filters, (1, time1_kernel), 
                    stride=(1, time1_stride))),
            ('time1_dropout', nn.Dropout(time1_dropout)),
            ('relu', nn.ReLU()),
            ('same_padding',
                nn.ConstantPad2d(
                    compute_same_pad2d(
                        (1, compute_out_size(n_samples, time1_kernel, stride=time1_stride)), 
                        (1, time2_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('time2_layer', 
                nn.Conv2d(n_time1_filters, n_time2_filters, (1, time2_kernel), 
                stride=(1, 1))),
            ('time2_dropout', nn.Dropout(time2_dropout)),
            ('flatten', nn.Flatten()),
            ('fc_layer', nn.Linear(
                n_time2_filters*compute_out_size(n_samples, time1_kernel, stride=time1_stride),
                n_classes))
        ]))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        _narrow_normal_weight_zero_bias(self)
        nn.init.ones_(self.model[0].weight)
        # MATLAB uses xavier_uniform_ with varaiance 2/(input+output)
        # perhaps this is a mistake in Help document
        nn.init.xavier_normal_(self.model[-1].weight, gain=1)

    def forward(self, X):
        # X: (n_batch, n_bands, n_channels, n_samples)
        out = self.model(X)
        return out
