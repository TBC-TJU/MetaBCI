# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet

# 使用MetaBCI的EEGNet模型
def train_classifier(X, y, num_channels, num_features, test_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = torch.tensor(X_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.double)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 初始化EEGNet模型
    model = EEGNet(n_channels=num_channels, n_samples=num_features, n_classes=2)

    # 设定优化器的学习率
    model.set_params(optimizer__lr=0.001)

    # 训练模型
    model.fit(X_train.numpy(), y_train.numpy(), epochs=200)

    # 在测试集上评估模型
    y_pred = model.predict(X_test.numpy())
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Final Test Results - Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return model

def evaluate_classifier(model, X, y):
    X = torch.tensor(X, dtype=torch.double)
    y = torch.tensor(y, dtype=torch.long)
    y_pred = model.predict(X.numpy())
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    return accuracy, precision, recall, f1

def predict_with_score(model, X):
    X = torch.tensor(X, dtype=torch.double)
    y_pred = model.predict(X.numpy())
    y_proba = model.predict_proba(X.numpy())
    return y_pred, y_proba[:, 1]  # 获取抑郁概率得分














