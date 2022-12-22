# -*- coding: utf-8 -*-
#
# Authors: zhaowei
# Date: 2022/10/27
# License: MIT License
"""
TUNERL Datasets

Weibo2014, Liang2020, Wei2020
"""
import os
from typing import Union, Optional, Dict
from pathlib import Path
from mne.io import read_raw_cnt
from mne.channels import make_standard_montage
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.utils.channels import upper_ch_names

filepath_mi = "data\\mi"
# 数据的相对路径
filepath_ssvep = "data\\ssvep"
MetaBCIData_URL = {
    'imagery': os.path.join(os.path.dirname(__file__), filepath_mi),
    'ssvep': os.path.join(os.path.dirname(__file__), filepath_ssvep)
}


class MetaBCIData(BaseDataset):
    _EVENTS = {
        'imagery': {
            "left_hand": (1, (0, 4)),
            "right_hand": (2, (0, 4)),
            "both_hands": (3, (0, 4)),
        },
        'ssvep': {
            '1': (1, 'a'), '2': (2, 'b'), '3': (3, 'c'), '4': (4, 'd'),
            '5': (5, 'e'), '6': (6, 'f'), '7': (7, 'g'), '8': (8, 'h'),
            '9': (9, 'i'), '10': (10, 'j'), '11': (11, 'k'), '12': (12, 'l'),
            '13': (13, 'm'), '14': (14, 'n'), '15': (15, 'o'), '16': (16, 'p'),
            '17': (17, 'q'), '18': (18, 'r'), '19': (19, 's'), '20': (20, 't'),
        }
    }

    _CHANNELS = {
        'imagery': ['FC3', 'FCZ', 'FC4', 'C3', 'CZ',
                    'C4', 'CP3', 'CPZ', 'CP4'],
        'ssvep': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                  'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
                  'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
    }

    def __init__(self, subjects, srate, paradigm, pattern='imagery'):
        self.pattern = pattern
        self.subjects = subjects
        self.srate = srate
        self.paradigm = paradigm

        super().__init__(
            dataset_code="Brainon",
            subjects=self.subjects,
            events=self._EVENTS[self.pattern],
            channels=self._CHANNELS[self.pattern],
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
            raise ValueError('Invalid subject {:d} given'.format(subject))

        if self.pattern == 'imagery':
            runs = list(range(1, 3))
        elif self.pattern == 'p300':
            runs = list(range(1, 4))
        elif self.pattern == 'ssvep':
            runs = list(range(1, 2))

        base_url = MetaBCIData_URL[self.pattern]
        dests = []
        for sub in self.subjects:
            dests.append(['{:s}\\sub{:d}\\{:d}.cnt'.format(
                base_url, sub, run) for run in runs])
        return dests

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = None):
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_cnt(run_file,
                                   eog=['HEO', 'VEO'],
                                   ecg=['EKG'], emg=['EMG'],
                                   misc=[32, 42, 59, 63],
                                   preload=True)
                raw = upper_ch_names(raw)
                raw = raw.pick_types(eeg=True, stim=True,
                                     selection=self.channels)
                raw.set_montage(montage)

                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
