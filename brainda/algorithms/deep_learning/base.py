# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping

def compute_out_size(input_size: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1):
    return int((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = ((stride-1)*input_size - stride + dilation*(kernel_size-1) + 1)
    return (all_padding//2, all_padding-all_padding//2)

def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0])
    lr = compute_same_pad1d(input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1])
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

def _adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        if not hasattr(self, 'num_samples_tracked'):
            self.num_samples_tracked = 0
            self.running_mean.data.zero_()
            self.running_var.data.zero_()
            self.running_var.data.fill_(1)
        k = len(inputs[0])
        self.num_samples_tracked += k
        
        module_name = self.__class__.__name__
        if 'BatchNorm1d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(input[0], dim=(0, 2))
        elif 'BatchNorm2d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif 'BatchNorm3d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))
        
        # see https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X
        d = mean - self.running_mean.data
        self.running_mean.data.add_(d*k/self.num_samples_tracked)
        self.running_var.data.add_(
            (var-self.running_var.data)*k/self.num_samples_tracked + torch.square(d)*k*(self.num_samples_tracked-k)/(self.num_samples_tracked**2)
        )
        
    if old_training_state:
        self.train()
        
def _global_adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        module_name = self.__class__.__name__
        if 'BatchNorm1d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(inputs[0], dim=(0, 2))
        elif 'BatchNorm2d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif 'BatchNorm3d' in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))
            
        self.running_mean.data.zero_()
        self.running_mean.data.add_(mean)
        self.running_var.data.zero_()
        self.running_var.data.add_(var)
    if old_training_state:
        self.train()
        
def adaptive_batch_norm(model, use_global=False):
    # register pre_forward_hook
    handles = []
    hook = _global_adabn_pre_forward_hook if use_global else _adabn_pre_forward_hook
    for module in model.modules():
        if 'BatchNorm' in module.__class__.__name__:
            handles.append(
                module.register_forward_pre_hook(hook))
    return model, handles

def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot uniform/xavier initialization, and setting biases to zero. 
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def _narrow_normal_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with 
    narrow normal distribution N(0, 0.01).
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class NeuralNetClassifierNoLog(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super(NeuralNetClassifier, self).get_loss(y_pred, y_true, *args, **kwargs)

    def fit(self, X, y, **fit_params):
        net = super(NeuralNetClassifier, self).fit(X, y, **fit_params)
        callbacks = OrderedDict(net.callbacks)
        if 'checkpoint' in callbacks:
            net.load_params(
                checkpoint=callbacks['checkpoint'])
        return net


class SkorchNet:
    def __init__(self, module):
        self.module = module
    
    def __call__(self, *args, **kwargs):
        model = self.module(*args, **kwargs)
        net = NeuralNetClassifierNoLog(model,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            batch_size=128, 
            lr=1e-2, 
            max_epochs=300,
            device="cpu",
            train_split=ValidSplit(0.2, stratified=True),
            iterator_train__shuffle=True,
            callbacks=[
                ('train_acc', EpochScoring('accuracy', 
                                            name='train_acc', 
                                            on_train=True, 
                                            lower_is_better=False)),
                ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=300 - 1)),
                ('estoper', EarlyStopping(patience=50)),
                ('checkpoint', Checkpoint(dirname="checkpoints/{:s}".format(str(id(model))))),
            ],
            verbose=True)
        return net