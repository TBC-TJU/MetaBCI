# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import os
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
from scipy.io import loadmat
from mne import create_info
from mne.io import RawArray
import numpy as np
from metabci.brainda.datasets.base import BaseDataset

class CustomRaw(RawArray):
    def __init__(self, data, info):
        super().__init__(data, info)

filepath_data = "E:\\BaiduNetdiskDownload\\Meta\\16alldata_xaunwu"

class MetaBCIData(BaseDataset):
    _EVENTS = {
        'default': {
            'rest': (1, (0, 2))
        }
    }

    def __init__(self, subjects, srate, paradigm, pattern='default'):
        self.pattern = pattern
        self.subjects = subjects
        self.srate = srate
        self.paradigm = paradigm

        super().__init__(
            dataset_code="CustomEEG",
            subjects=self.subjects,
            events=self._EVENTS[self.pattern],
            channels=[],  # 通道将在数据文件中获取
            srate=self.srate,
            paradigm=self.paradigm
        )

    def data_path(self,
                  subject: Union[str, int],
                  path: Optional[Union[str, Path]] = None,
                  force_update: bool = False,
                  update_path: Optional[bool] = None,
                  proxies: Optional[Dict[str, str]] = None,
                  verbose: Optional[Union[bool, str, int]] = None):

        if subject not in self.subjects:
            raise ValueError(f'Invalid subject {subject} given')

        if path is None:
            path = Path(filepath_data)
        
        if subject.startswith('H'):
            return [[path / f"H{subject[1:]}new.mat"]]
        elif subject.startswith('D'):
            return [[path / f"D{subject[1:]}new.mat"]]
        else:
            return [[path / f"{subject}.mat"]]

    def _get_single_subject_data(self,
                                 subject: Union[str, int],
                                 verbose: Optional[Union[bool, str, int]] = None):
        data_path = self.data_path(subject)[0][0]
        
        # 确认数据路径
        print(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        mat = loadmat(data_path)
        
        # 确认数据结构
        print(f"Keys in the MAT file for subject {subject}: {mat.keys()}")
        
        # 获取 EEG 数据
        eeg_data = mat['EEG'][0, 0]
        data = np.array(eeg_data['data'], dtype=np.float64)  # 确保数据类型正确
        srate = eeg_data['srate'][0, 0]
        channels = eeg_data['chanlocs'][0]
        channel_names = [ch['labels'][0] for ch in channels]
        
        info = create_info(ch_names=channel_names, sfreq=srate, ch_types='eeg')
        raw = CustomRaw(data, info)
        return {f'session_1': {f'run_1': raw}}

    def get_data(self, subjects: List[Union[int, str]], verbose: Optional[Union[bool, str, int]] = None):
        data = dict()
        for subject in subjects:
            if subject not in self.subjects:
                raise ValueError(f"Invalid subject {subject} given")
            data[subject] = self._get_single_subject_data(subject)
        return data

