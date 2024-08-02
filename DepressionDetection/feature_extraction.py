# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
from mne.time_frequency import psd_array_multitaper

def extract_features(raw, num_features=1000):
    data = raw.get_data()
    psds, freqs = psd_array_multitaper(data, sfreq=raw.info['sfreq'], adaptive=True, normalization='full', verbose=0)
    psds = 10 * np.log10(psds)  # 将功率谱密度转换为dB
    
    # 确保返回的特征数量为模型期望的通道数和固定的特征数量
    num_channels, actual_num_features = psds.shape
    if num_channels != 16:
        psds = psds[:16, :]  # 截取或补齐为 16 个通道

    # 截取或填充特征数量为固定值
    if actual_num_features < num_features:
        psds = np.pad(psds, ((0, 0), (0, num_features - actual_num_features)), 'constant')
    else:
        psds = psds[:, :num_features]

    avg_power = np.mean(psds, axis=1)  # 计算平均功率

    return psds, freqs, avg_power













