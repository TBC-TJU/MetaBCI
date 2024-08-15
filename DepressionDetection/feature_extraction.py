# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
from mne.time_frequency import psd_array_multitaper
from metabci.brainda.algorithms.manifold.rpa import get_rescale, rescale  # 导入rpa中的函数

def extract_features(raw, num_features=1000):
    data = raw.get_data()

    # 计算功率谱密度（PSD）
    psds, freqs = psd_array_multitaper(data, sfreq=raw.info['sfreq'], adaptive=True, normalization='full', verbose=0)
    psds = 10 * np.log10(psds)  

    # 使用get_rescale函数计算出协方差矩阵的缩放因子
    M, scale = get_rescale(data)

    # 使用rescale函数对数据进行缩放
    scaled_data = rescale(data, M, scale)

    num_channels, actual_num_features = psds.shape
    if num_channels != 16:
        psds = psds[:16, :]  

    if actual_num_features < num_features:
        padding = num_features - actual_num_features
        psds = np.pad(psds, ((0, 0), (0, padding)), 'constant')
    else:
        psds = psds[:, :num_features]

    avg_power = np.mean(psds, axis=1)  

    # 调整scaled_data形状以匹配psds形状
    scaled_data_shape = scaled_data.shape
    if len(scaled_data_shape) == 2:
        if scaled_data_shape[1] < num_features:
            padding = num_features - scaled_data_shape[1]
            scaled_data = np.pad(scaled_data, ((0, 0), (0, padding)), 'constant')
        elif scaled_data_shape[1] > num_features:
            scaled_data = scaled_data[:, :num_features]

    return psds, freqs, avg_power
















