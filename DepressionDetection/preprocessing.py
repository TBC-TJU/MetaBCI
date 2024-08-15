# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
from mne.preprocessing import ICA
from metabci.brainda.algorithms.feature_analysis.time_analysis import TimeAnalysis
import pandas as pd
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainflow.amplifiers import Marker
from metabci.brainflow.logger import get_logger

def preprocess_data(raw):
    # 日志记录
    logger = get_logger('preprocessing')
    logger.info('Starting preprocessing')

    # 滤波
    raw.filter(1., 40., fir_design='firwin')
    logger.info('Filtering done')

    # 重参考
    raw.set_eeg_reference('average', projection=True)
    logger.info('Reference set')

    # ICA 去伪迹
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.apply(raw)
    logger.info('ICA artifact removal done')

    # 创建meta对象，用pandas DataFrame存储事件信息
    event_label = 'rest'
    n_samples = raw.get_data().shape[1]
    meta = pd.DataFrame({
        'event': [event_label] * n_samples 
    })

    # 使用BrainFlow的Marker类创建环形缓冲区
    marker = Marker(interval=[0.1, 2.1], srate=raw.info['sfreq'], events=[1])
    
    # 创建DummyDataset类，继承自BaseDataset
    class DummyDataset(BaseDataset):
        def __init__(self, srate):
            self.srate = srate

        def _get_single_subject_data(self, subject):
            return None

        def data_path(self, subject, path=None, force_update=False, update_path=None, proxies=None, verbose=None):
            return None

    dummy_dataset = DummyDataset(srate=raw.info['sfreq'])

    data = raw.get_data()

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0) 
    
    if data.shape[0] == 1:  
        data = np.expand_dims(data, axis=0) 
    
    # 分批处理数据（避免内存溢出）
    enhanced_data_list = []
    for i in range(data.shape[2]):  # 逐个通道处理
        single_channel_data = data[:, :, i:i+1, :] 
        try:
            # 调用TimeAnalysis进行时间分析
            time_analysis = TimeAnalysis(data=single_channel_data, meta=meta, dataset=dummy_dataset, event='rest')
            enhanced_data = time_analysis.stacking_average(data=single_channel_data, _axis=0)
            enhanced_data_list.append(enhanced_data)
            # 在处理过程中使用BrainFlow的Marker类截取数据
            if marker(event=1):
                epoch_data = marker.get_epoch()
                logger.info(f"Epoch data retrieved: {epoch_data.shape}")
        except MemoryError as e:
            pass
        except IndexError as e:
            pass

    # 将增强后的数据组合
    if enhanced_data_list:
        enhanced_data_combined = np.concatenate(enhanced_data_list, axis=1)
        raw._data = enhanced_data_combined  

    logger.info('Preprocessing complete')
    return raw






















 

