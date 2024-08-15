# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

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

@SkorchNet 
class CNN_GRU_Attn(nn.Module):
    """
    CNN_GRU_Attn is a neural network specifically designed for EEG-based Depression Detection.
    The network consists of convolutional layers, a bidirectional GRU layer, and an attention mechanism.
    
    author: ChenXiaoxin <Chenxx@emails.bjut.edu.cn>
    
    Created on: 2024-08-01
    
    Parameters
    ----------
    n_channels: int
        Number of channels for the input signal.
    n_samples: int
        Number of sampling points of the input signal.
    n_classes: int
        Number of classes of input signals to be classified.

    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> num_classes = 2
    >>> model = CNN_GRU_Attn(n_channels=16, n_samples=100, n_classes=num_classes)
    >>> model.fit(X_train, y_train)

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    """

    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__() 
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.gru = nn.GRU(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, n_classes)
        
        self._reset_parameters()

    def attention_net(self, gru_output):
        attn_weights = torch.softmax(self.attention(gru_output), dim=1)
        attn_output = torch.sum(gru_output * attn_weights, dim=1)
        return attn_output

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool1d(x, kernel_size=2)
        x = self.conv2(x)
        x = torch.relu(x)
        x, _ = self.gru(x.permute(0, 2, 1))
        x = self.attention_net(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def fit(self, X_train, y_train, epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
        return predicted

