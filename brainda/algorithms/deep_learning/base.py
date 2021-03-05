# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_out_size(input_size: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1):
    return int((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = ((stride-1)*input_size - stride + dilation*(kernel_size-1) + 1)
    return (all_padding//2, all_padding-all_padding//2)

def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0])
    lr = compute_same_pad1d(input_size[1], kernel_size[1], stride=stride[0], dilation=dilation[1])
    return [*lr, *ud]

class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w

class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w    


