# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023/1/10
# License: MIT License

"""
aVEP datasets

"""

import os
import numpy as np
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

import mne.channels
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
from metabci.brainda.utils.channels import upper_ch_names

filepath = "/Users/meijie/Documents/Data/edf_data"


class XuaVEPDataset(BaseTimeEncodingDataset):
    _MINOR_EVENTS = {
        "left-right": (1, (0.05, 0.45)),
        "right-left": (2, (0.05, 0.45)),
    }

    _EVENTS = {
        "A": (51, (0, 7.6)),
        "B": (52, (0, 7.6)),
        "C": (53, (0, 7.6)),
        "D": (54, (0, 7.6)),
        "E": (61, (0, 7.6)),
        "F": (62, (0, 7.6)),
        "G": (63, (0, 7.6)),
        "H": (64, (0, 7.6)),
        "I": (71, (0, 7.6)),
        "J": (72, (0, 7.6)),
        "K": (73, (0, 7.6)),
        "L": (74, (0, 7.6)),
        "M": (81, (0, 7.6)),
        "N": (82, (0, 7.6)),
        "O": (83, (0, 7.6)),
        "P": (84, (0, 7.6)),
        "Q": (91, (0, 7.6)),
        "R": (92, (0, 7.6)),
        "S": (93, (0, 7.6)),
        "T": (94, (0, 7.6)),
        "U": (101, (0, 7.6)),
        "V": (102, (0, 7.6)),
        "W": (103, (0, 7.6)),
        "X": (104, (0, 7.6)),
        "Y": (111, (0, 7.6)),
        "Z": (112, (0, 7.6)),
        "1": (113, (0, 7.6)),
        "2": (114, (0, 7.6)),
        "3": (121, (0, 7.6)),
        "4": (122, (0, 7.6)),
        "5": (123, (0, 7.6)),
        "6": (124, (0, 7.6))
    }

    _ALPHA_CODE = {
        "A": [1, 2, 2, 1, 1],
        "B": [1, 2, 1, 2, 1],
        "C": [1, 2, 1, 1, 1],
        "D": [1, 1, 2, 2, 2],
        "E": [2, 1, 2, 1, 2],
        "F": [2, 1, 1, 2, 1],
        "G": [1, 1, 2, 1, 1],
        "H": [2, 2, 1, 2, 1],
        "I": [2, 1, 1, 2, 2],
        "J": [1, 1, 1, 2, 1],
        "K": [1, 1, 2, 1, 2],
        "L": [2, 2, 1, 1, 2],
        "M": [2, 1, 2, 2, 1],
        "N": [1, 2, 2, 1, 2],
        "O": [2, 2, 2, 1, 2],
        "P": [1, 1, 2, 2, 1],
        "Q": [2, 2, 1, 1, 1],
        "R": [1, 1, 1, 1, 2],
        "S": [2, 1, 2, 1, 1],
        "T": [1, 2, 1, 2, 2],
        "U": [2, 2, 2, 1, 1],
        "V": [2, 1, 1, 1, 2],
        "W": [2, 1, 1, 1, 1],
        "X": [1, 1, 1, 2, 2],
        "Y": [1, 2, 2, 2, 1],
        "Z": [2, 2, 2, 2, 1],
        "1": [2, 2, 1, 2, 2],
        "2": [2, 2, 2, 2, 2],
        "3": [1, 2, 1, 1, 2],
        "4": [2, 1, 2, 2, 2],
        "5": [1, 1, 1, 1, 1],
        "6": [1, 2, 2, 2, 2]
    }

    _ENCODE_LOOP = 6

    _CHANNELS = [
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
        'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3',
        'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1',
        'OZ', 'O2', 'CB2'
    ]

    def __init__(self, paradigm='aVEP'):
        super().__init__(
            dataset_code="Xu_aVEP",
            subjects=list(range(1, 29)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=200,
            paradigm=paradigm,
            minor_events=self._MINOR_EVENTS,
            encode=self._ALPHA_CODE,
            encode_loop=self._ENCODE_LOOP
        )

    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ):
        if subject not in self.subjects:
            raise ValueError('Invalid subject {:d} given'.format(subject))

        runs = list(range(1, 7))
        sessions = list(range(1))
        base_url = filepath
        subject = cast(int, subject)
        if subject < 10:
            sub_name = '0' + str(subject)
        else:
            sub_name = str(subject)
        # dests.append(['{:s}\\Sub{:s}\\session_0{:s}.edf'.format(
        #     base_url, sub_name, str(run)) for run in runs])
        # append data file path
        # data_path.append(['{:s}/Sub{:s}/session_0{:s}.edf'.format(
        #     base_url, sub_name, str(run)) for run in runs])
        # # append event file path
        # event_path.append(['{:s}/Sub{:s}/session_0{:s}_events.edf'.format(
        #     base_url, sub_name, str(run)) for run in runs])
        sessions_dests = []
        for session in sessions:
            dests = []
            for run in runs:
                data_path = '{:s}/Sub{:s}/session_0{:s}.edf'.format(base_url, sub_name, str(run))
                event_path = '{:s}/Sub{:s}/session_0{:s}_events.edf'.format(base_url, sub_name, str(run))
                dests.append((data_path, event_path))
            sessions_dests.append(dests)
        return sessions_dests

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = False
    ):
        dests = self.data_path(subject)
        # montage = make_standard_montage('standard_1005')
        montage = mne.channels.read_custom_montage(os.path.join(filepath, '64-channels.loc'))
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for idx_sess, run_files_path in enumerate(dests):
            runs = dict()
            raw_temp = []
            for idx_run, run_file in enumerate(run_files_path):
                raw = read_raw_edf(run_file[0], preload=True)
                events = mne.read_events(run_file[1])
                stim_chan = np.zeros((1, raw.__len__()))
                for index in range(events.shape[0]):
                    stim_chan[0, events[index, 0]] = events[index, 2]
                stim_chan_name = ['STI 014']
                stim_chan_type = "stim"
                stim_info = mne.create_info(
                    ch_names=stim_chan_name,
                    ch_types=stim_chan_type,
                    sfreq=self.srate
                )
                stim_raw = mne.io.RawArray(
                    data=stim_chan,
                    info=stim_info
                )
                # add the stim_chan to data raw object
                raw.add_channels([stim_raw])
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                raw_temp.append(raw)
            raw_temp[0].append(raw_temp[1])
            raw_temp[2].append(raw_temp[3])
            raw_temp[4].append(raw_temp[5])
            runs['run_1'] = raw_temp[0]
            runs['run_2'] = raw_temp[2]
            runs['run_3'] = raw_temp[4]
            sess['session_{:d}'.format(idx_sess)] = runs
        return sess
