# -*- coding: utf-8 -*-
"""
China BCI Competition.
"""
from brainda.utils.download import mne_data_path
import os
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
from mne import create_info
from mne.io import Raw, read_raw_cnt, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

# no available links now
CBCIC2019001_URL = 'file:///CBCIC2019001'
CBCIC2019004_URL = 'file:///CBCIC2019004'


class CBCIC2019001(BaseDataset):
    """2019 China BCI competition Dataset for MI in preliminary contest A/B.

    Motor imagery dataset from China BCI competition in 2019.

    This dataset contains EEG recordings from 18 subjects, performing 2 or 3 tasks 
    of motor imagery (left hand, right hand or feet). Data have been recored at 1000hz 
    with 64 electrodes (59 in use except ECG, HEOR, HEOL, VEOU, VEOL channels) by
    an neuracle EEG amplifier.

    """

    _EVENTS = {
        "left_hand": (1, (1.5, 5.5)),
        "right_hand": (2, (1.5, 5.5)),
        "feet": (3, (1.5, 5.5)),
        # "open_eye_relax": (7, (0, 60)),
        # "close_eye_relax": (8, (0, 60))
    }

    _CHANNELS = [
        'FPZ','FP1','FP2','AF3','AF4','AF7','AF8',
        'FZ','F1','F2','F3','F4','F5','F6','F7','F8',
        'FCZ','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8',
        'CZ','C1','C2','C3','C4','C5','C6',
        'T7','T8','CP1','CP2','CP3','CP4','CP5','CP6',
        'TP7','TP8','PZ','P3','P4','P5','P6','P7','P8',
        'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
        'OZ','O1','O2'
    ]
    
    def __init__(self):
        super().__init__(
            dataset_code="cbcic2019001",
            subjects=list(range(1, 19)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm='imagery')

    def data_path(self, 
            subject: Union[str, int], 
            path: Optional[Union[str, Path]] = None, 
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None) -> List[List[Union[str, Path]]]:

        if subject not in self.subjects:
            raise(ValueError("Invalid subject id"))

        if subject in [6, 14, 15, 18]:
            file_name = "T{:02d}01T.mat".format(subject)
        else:
            file_name = "B{:02d}01T.mat".format(subject)

        url = "{:s}/{:02d}/{:s}".format(CBCIC2019001_URL, subject, file_name)
        dests = [
            [
                mne_data_path(url, 'cbcic', 
                    path=path, proxies=proxies, force_update=force_update, update_path=update_path)
            ]
        ]
        return dests
     
    def _get_single_subject_data(self, subject: Union[str, int], 
            verbose: Optional[Union[bool, str, int]] = None) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw_mat = loadmat(run_file)['EEG']
                epoch_data = raw_mat['data'][:-5] * 1e-6
                stim = np.zeros((1, epoch_data.shape[-1]))
                for event in raw_mat['event']:
                    stim[0, int(event['latency'])-1] = int(event['type'])
                data = np.concatenate((epoch_data, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in  self._CHANNELS]
                ch_types = ['eeg']*len(ch_names)
                ch_names = ch_names + ['STI 014']
                ch_types = ch_types + ['stim']

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)
                
                raw = RawArray(data=data, info=info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess


class CBCIC2019004(BaseDataset):
    """2019 China BCI competition Dataset for MI in final competition.

    Motor imagery dataset from China BCI competition in 2019.

    This dataset contains EEG recordings from 18 subjects, performing 2 or 3 tasks 
    of motor imagery (left hand, right hand or feet). Data have been recored at 1000hz 
    with 64 electrodes (59 in use except ECG, HEOR, HEOL, VEOU, VEOL channels) by
    an neuracle EEG amplifier.

    """

    _EVENTS = {
        "left_hand": (1, (0, 4)),
        "right_hand": (2, (0, 4)),
    }

    _CHANNELS = [
        'FPZ','FP1','FP2','AF3','AF4','AF7','AF8',
        'FZ','F1','F2','F3','F4','F5','F6','F7','F8',
        'FCZ','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8',
        'CZ','C1','C2','C3','C4','C5','C6',
        'T7','T8','CP1','CP2','CP3','CP4','CP5','CP6',
        'TP7','TP8','PZ','P3','P4','P5','P6','P7','P8',
        'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
        'OZ','O1','O2'
    ]
    
    def __init__(self):
        super().__init__(
            dataset_code="cbcic2019004",
            subjects=list(range(1, 7)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm='imagery')

    def data_path(self, 
            subject: Union[str, int], 
            path: Optional[Union[str, Path]] = None, 
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None) -> List[List[Union[str, Path]]]:

        if subject not in self.subjects:
            raise(ValueError("Invalid subject id"))

        runs = []
        for i in range(1, 5):
            url = "{:s}/{:02d}/block{:d}.mat".format(CBCIC2019004_URL, subject, i)
            runs.append(
                mne_data_path(url, 'cbcic', 
                    path=path, proxies=proxies, force_update=force_update, update_path=update_path)
            )

        dests = [
            runs
        ]
        return dests
     
    def _get_single_subject_data(self, subject: Union[str, int], 
            verbose: Optional[Union[bool, str, int]] = None) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw_mat = loadmat(run_file)
                epoch_data = raw_mat['data'][:-6] * 1e-6
                stims = raw_mat['data'][-1][np.newaxis, :]
                data = np.concatenate((epoch_data, stims), axis=0)

                ch_names = [ch_name.upper() for ch_name in  self._CHANNELS]
                ch_types = ['eeg']*len(ch_names)
                ch_names = ch_names + ['STI 014']
                ch_types = ch_types + ['stim']

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)
                
                raw = RawArray(data=data, info=info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess

