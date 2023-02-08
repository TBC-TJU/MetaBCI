# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023/02/06
# License: MIT License

from typing import Dict
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
import numpy as np
from numpy import ndarray


def concat_trials(x: Dict, y: Dict):
    x_temp = []
    y_temp = []
    if len(list(x.keys())) != len(list(y.keys())):
        raise KeyError('keys number of x and y is not equal')
    for key in x.keys():
        x_temp.extend(x[key])
        y_temp.extend(y[key])

    x_concat = np.concatenate(x_temp, axis=0)
    y_concat = np.concatenate(y_temp, axis=0)

    return x_concat, y_concat


class TimeDecodeTool:
    def __init__(self, dataset: BaseTimeEncodingDataset, feature_operation: str = 'sum'):
        # Get minor event from the dataset
        minor_events = dataset.minor_events
        minor_class = list()
        for event in minor_events.values():
            minor_class.append(event[0])
        minor_class.sort()
        self.minor_class = np.array(minor_class)
        self.encode_map = dataset.encode
        self.encode_loop = dataset.encode_loop
        self.feature_operation = feature_operation

    def _trial_feature_split(self, key: str, feature: ndarray):
        key_encode = self.encode_map[key]
        key_encode_len = len(key_encode)
        if key_encode_len * self.encode_loop != feature.shape[0]:
            raise ValueError('Epochs in the test trial does not same '
                             'as the presetting parameter in dataset')
        # create a space for storage feature
        feature_storage = np.zeros((self.encode_loop, key_encode_len, *feature.shape[1:]))
        for row in range(self.encode_loop):
            for col in range(key_encode_len):
                feature_storage[row][col] = feature[row * key_encode_len + col, :]

        return key, feature_storage

    def _features_operation(self, feature_storage: ndarray):
        if self.feature_operation == 'sum':
            sum_feature = np.sum(feature_storage, axis=0, keepdims=False)
            return sum_feature

    def _predict(self, features: ndarray):
        predict_labels = self.minor_class[np.argmax(features, axis=-1)]
        return predict_labels

    def _find_command(self, predict_labels: ndarray):
        for key, value in self.encode_map.items():
            if np.array_equal(np.array(value), predict_labels):
                return key
        return None

    def decode(self, key: str, feature: ndarray):
        alpha_key, feature_storage = self._trial_feature_split(key, feature)
        merge_features = self._features_operation(feature_storage)
        predict_labels = self._predict(merge_features)
        command = self._find_command(predict_labels)
        return command
