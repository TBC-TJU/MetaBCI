# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/02/22
# License: MIT License
"""
GigaDb Motor imagery dataset.
"""
import os
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
from mne import create_info
from mne.io import Raw, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

GIGA_URL = 'ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/'


class Cho2017(BaseDataset):
    """Motor Imagery dataset from Cho et al 2017.

    Dataset from the paper [1]_.

    **Dataset Description**

    We conducted a BCI experiment for motor imagery movement (MI movement)
    of the left and right hands with 52 subjects (19 females, mean age ± SD
    age = 24.8 ± 3.86 years); Each subject took part in the same experiment,
    and subject ID was denoted and indexed as s1, s2, …, s52.
    Subjects s20 and s33 were both-handed, and the other 50 subjects
    were right-handed.

    EEG data were collected using 64 Ag/AgCl active electrodes.
    A 64-channel montage based on the international 10-10 system was used to
    record the EEG signals with 512 Hz sampling rates.
    The EEG device used in this experiment was the Biosemi ActiveTwo system.
    The BCI2000 system 3.0.2 was used to collect EEG data and present
    instructions (left hand or right hand MI). Furthermore, we recorded
    EMG as well as EEG simultaneously with the same system and sampling rate
    to check actual hand movements. Two EMG electrodes were attached to the
    flexor digitorum profundus and extensor digitorum on each arm.

    Subjects were asked to imagine the hand movement depending on the
    instruction given. Five or six runs were performed during the MI
    experiment. After each run, we calculated the classification
    accuracy over one run and gave the subject feedback to increase motivation.
    Between each run, a maximum 4-minute break was given depending on
    the subject's demands.

    References
    ----------

    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034
    """

    _EVENTS = {
        "left_hand": (1, (0, 3)), 
        "right_hand": (2, (0, 3)), 
    }

    _CHANNELS = [
        'FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
        'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
        'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ',
        'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
        'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 
    ]

    def __init__(self):
        super().__init__(
            dataset_code='cho2017', 
            subjects=list(range(1, 53)),
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=512,
            paradigm='imagery'
        )

    def data_path(self, 
            subject: Union[str, int], 
            path: Optional[Union[str, Path]] = None, 
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise(ValueError("Invalid subject id"))

        url = '{:s}s{:02d}.mat'.format(GIGA_URL, subject)
        file_dest = mne_data_path(url, self.dataset_code, 
            path=path, proxies=proxies, force_update=force_update, update_path=update_path)
        dests = [
            [file_dest]
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
                raw_mat = loadmat(run_file)['eeg']
                eeg_data_l = np.concatenate((raw_mat['imagery_left'] * 1e-6, raw_mat['imagery_event'].reshape((1, -1))), axis=0)
                eeg_data_r = np.concatenate((raw_mat['imagery_right'] * 1e-6, raw_mat['imagery_event'].reshape((1, -1))*2),
                axis=0)

                data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r])
                ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'STI 014']
                ch_types = ['eeg']*len(self._CHANNELS) + ['emg']*4 + ['stim']

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate
                )
                raw = RawArray(
                    data=data, info=info, verbose=verbose
                )
                raw = upper_ch_names(raw)
                raw.set_montage(montage)

                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
