# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023/02/06
# License: MIT License

from typing import Dict
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from typing import Optional, Union, Dict
from numpy.random import RandomState
import copy
import mne


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

    def _features_operation(self, feature_storage: ndarray, fold_num=6):
        if fold_num > np.shape(feature_storage)[0]:
            raise ValueError("The number of trial stacks cannot exceeds %d" % np.shape(feature_storage)[0])
        if self.feature_operation == 'sum':
            sum_feature = np.sum(feature_storage[0:fold_num], axis=0, keepdims=False)
            return sum_feature

    def _predict(self, features: ndarray):
        predict_labels = self.minor_class[np.argmax(features, axis=-1)]
        return predict_labels

    def _predict_p300(self, features: ndarray):
        code_len = features.shape[0]
        half_len = int(code_len/2)
        predict_row = np.argmax(features[:half_len, -1])
        predict_col = np.argmax(features[half_len:, -1])+6
        predict_labels = np.ones_like(self.minor_class, dtype=int)
        predict_labels[predict_row] = 2
        predict_labels[predict_col] = 2
        return predict_labels

    def _find_command(self, predict_labels: ndarray):
        for key, value in self.encode_map.items():
            if np.array_equal(np.array(value), predict_labels):
                return key
        return None

    def decode(self, key: str, feature: ndarray, fold_num=6, paradigm='avep'):
        if feature.ndim < 2:
            feature = feature[:, np.newaxis]
        alpha_key, feature_storage = self._trial_feature_split(key, feature)
        merge_features = self._features_operation(feature_storage, fold_num)
        predict_labels = []
        if paradigm == 'avep':
            predict_labels = self._predict(merge_features)
        elif paradigm == 'p300':
            predict_labels = self._predict_p300(merge_features)
        command = self._find_command(predict_labels)
        return command

    def target_calibrate(self, y, key):
        y_tar = []
        for i in range(len(y)):
            character = key.values[i]
            target_id = np.where(
                np.array(self.encode_map[character]) == 2)[0]+1
            target_loc = []
            event = y[i].copy()
            for j in target_id:
                target_loc = np.append(target_loc, np.where(event == j))
            target_loc = np.array(target_loc, dtype=int)

            event[:] = 1
            event[target_loc] = 2
            y_tar.append(event)
        return y_tar

    def resample(self, x, fs_old, fs_new, axis=None):
        if axis is None:
            axis = x.ndim-1
        down_factor = fs_old/fs_new
        x_1 = mne.filter.resample(x, down=down_factor, axis=axis)
        return x_1

    def epoch_sort(self, X, y):
        code_len = len(self.minor_class)
        X_sort = [[] for i in range(len(X))]
        Y_sort = [[] for i in range(len(y))]
        for char_i in range(len(X)):
            for loop_i in range(self.encode_loop):
                epoch_id = np.arange(loop_i*code_len, (loop_i+1)*code_len)
                y_i = y[char_i][epoch_id]
                x_i = X[char_i][epoch_id]

                id = np.argsort(y_i)
                x_sort = x_i[id, :, :]
                y_sort = y_i[id]
                X_sort[char_i].append(x_sort)
                Y_sort[char_i].append(y_sort)
            X_sort[char_i] = np.concatenate(X_sort[char_i], axis=0)
            Y_sort[char_i] = np.concatenate(Y_sort[char_i], axis=0)
        return X_sort, Y_sort
