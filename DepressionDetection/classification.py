# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CNNGRUModel(nn.Module):
    def __init__(self, num_channels, num_features):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, 2)  # 2 classes: Healthy and Depressed

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # 交换维度以适应 GRU 的输入要求
        x, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        x = self.fc(x)
        return x

def train_classifier(X, y, num_channels, num_features, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = CNNGRUModel(num_channels=num_channels, num_features=num_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = accuracy_score(y_test, predicted)
                precision = precision_score(y_test, predicted, average='macro')
                recall = recall_score(y_test, predicted, average='macro')
                f1 = f1_score(y_test, predicted, average='macro')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
                      f'Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, '
                      f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return model

def evaluate_classifier(model, X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y, predicted)
        precision = precision_score(y, predicted, average='macro')
        recall = recall_score(y, predicted, average='macro')
        f1 = f1_score(y, predicted, average='macro')
    return accuracy, precision, recall, f1

def predict_with_score(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        scores = torch.softmax(outputs, dim=1).numpy()[:, 1]  # 获取抑郁概率得分
    return predicted.numpy(), scores









