# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

"""
CNN_GRU_Attn network for EEG-based Depression Detection.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import scipy.io
import matplotlib.pyplot as plt
import itertools
import numpy as np

class CNN_GRU_Attn(nn.Module):
    """
    CNN_GRU_Attn is a neural network specifically designed for EEG-based Depression Detection.
    The network consists of convolutional layers, a bidirectional GRU layer, and an attention mechanism.
    
    author: Your Name <xmanwo@163.com>
    
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
    >>> model.forward(X)

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    """

    def __init__(self, n_channels, n_samples, n_classes):
        super(CNN_GRU_Attn, self).__init__()
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

# 数据加载和预处理
data = scipy.io.loadmat(r'/Users/huchenyang/Desktop/EEG Project/马卓学习资料/代码/melo/depression_psd.mat')
X = data['data'].reshape(7533, 16, 100)
y = data['datalable'].reshape(7533, 1)
X, y = shuffle(X, y, random_state=0)

mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

α = 0.9
A = int(len(X) * α)
x_train, x_test = X[:A], X[A:]
y_train, y_test = y[:A], y[A:]

# 转换为PyTorch张量
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

# 数据加载器
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 模型初始化
model = CNN_GRU_Attn(n_channels=16, n_samples=100, n_classes=2)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.005, alpha=0.9, eps=1e-6, weight_decay=0.01)

# 模型训练
num_epochs = 70
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# 模型评估
model.eval()
y_pred_list = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, y_pred = torch.max(outputs, 1)
        y_pred_list.extend(y_pred.numpy())

# 混淆矩阵
def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6.5, 5))
    classes = ['Depression', 'Health']
    lables = range(2)
    matrix = confusion_matrix(y_true, y_pred, labels=lables)
    cmap = plt.cm.Blues
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=7.5)
    plt.yticks(tick_marks, classes, fontsize=7.5)
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.title("Confusion matrix", fontsize=10)
    plt.ylabel("Real label", fontsize=9)
    plt.xlabel("Predict label", fontsize=9)
    plt.show()

# 评估和报告
y_test_np = y_test.numpy()
print(classification_report(y_test_np, y_pred_list))
show_confusion_matrix(y_test_np, y_pred_list)

