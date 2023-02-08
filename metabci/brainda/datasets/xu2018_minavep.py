# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023/1/10
# License: MIT License

"""
aVEP datasets

"""


import numpy as np
from typing import Union, Optional, Dict, cast
from pathlib import Path

import mne.channels
from mne.io import read_raw_cnt
from mne.channels import make_standard_montage
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
from metabci.brainda.utils.channels import upper_ch_names

# The filepath will be available when the dataset is uploaded
filepath = ""


class Xu2018MinaVep(BaseTimeEncodingDataset):
    """
    Dataset in:
    M. Xu, X. Xiao, Y. Wang, H. Qi, T. -P. Jung and D. Ming,
    "A Brain–Computer Interface Based on Miniature-Event-Related
    Potentials Induced by Very Small Lateral Visual Stimuli,"
    in IEEE Transactions on Biomedical Engineering,
    vol. 65, no. 5, pp. 1166-1175, May 2018,
    doi: 10.1109/TBME.2018.2799661.

    This study implemented a miniature aVEP-based BCI speller,
    and proposed a new scheme for BCI encoding. Thirty-two
    alphanumeric characters were arranged as a 4 × 8 matrix
    displayed on a computer screen and encoded by a new SCDMA
    scheme, in which the left and right lateral visual stimuli
    constituted two parallel spatial channels while two different
    lateral visual stimuli sequences made up the basic communication
    codes ‘0’ and ‘1’. Specifically, the ‘left-right’ stimulus sequence,
    which lasted 200 ms, was regarded as code ‘0’, while ‘right-left’
    stimulus was coded ‘1’. Thirty-two different code sequences were
    created using 5 bits in this study, which were arbitrarily
    allocated to different characters. Specifically, character
    A’ was encoded by ‘01100’. In spelling, the lateral visual
    stimuli would be presented simultaneously for all characters
    with different sequences. To obtain a reliable output, the
    same code sequence was repeated 6 times for the offline
    spelling and individually optimized times for the online
    spelling. The character specified to output in the offline
    spelling would be indicated by a star-shaped cue underneath
    for 0.8 seconds, which would be offset for another 0.2 seconds
    to wipe out the cue effect. There was a time interval of 0.2
    seconds with no stimulation between two successive sequences.

    EEG was recorded using a Neuroscan Synamps2 system with 64
    electrodes located in the positions following the 10/20 system.
    The reference electrode was put in the central area near Cz and
    the ground electrode was put on the frontal lobe. The recorded
    signals were bandpass-filtered at 0.1–100 Hz, notch-filtered at
    50 Hz, digitized at a rate of 1000 Hz and then stored in a computer.

    """
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
        'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
        'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
        'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
    ]

    def __init__(self, paradigm='aVEP'):
        super().__init__(
            dataset_code="Xu_aVEP_min_aVEP",
            subjects=list(range(1, 13)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm=paradigm,
            minor_events=self._MINOR_EVENTS,
            encode=self._ALPHA_CODE,
            encode_loop=self._ENCODE_LOOP
        )
        self.events_list = [value[0] for value in self._EVENTS.values()]
        self.events_key_map = {value[0]: key for key, value in self._EVENTS.items()}

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
            raise ValueError('Invalid subject {} given'.format(subject))

        runs = list(range(1, 7))
        sessions = list(range(1))
        base_url = filepath
        subject = cast(int, subject)
        sub_name = str(subject)
        sessions_dests = []
        for session in sessions:
            dests = []
            for run in runs:
                data_path = '{:s}/S{:s}/VEP_nophase_{:s}.cnt'.format(base_url, sub_name, str(run))
                dests.append(data_path)
            sessions_dests.append(dests)
        return sessions_dests

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = False
    ):
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for idx_sess, run_files_path in enumerate(dests):
            runs = dict()
            raw_temp = []
            for idx_run, run_file in enumerate(run_files_path):
                raw = read_raw_cnt(run_file,
                                   eog=['HEO', 'VEO'],
                                   ecg=['EKG'],
                                   emg=['EMG'],
                                   misc=[32, 42, 59, 63],
                                   preload=True)
                raw = upper_ch_names(raw)
                raw = raw.pick_types(eeg=True,
                                     stim=True,
                                     selection=self.channels)
                raw.set_montage(montage)
                stim_chan = np.zeros((1, raw.__len__()))
                # Convert annotation to event
                events, _ = \
                    mne.events_from_annotations(raw, event_id=(lambda x: int(x)))
                # Insert the event to the event channel
                for index in range(events.shape[0]):
                    if events[index, 2] in self.events_list:
                        stim_chan[0, events[index, 0]] = events[index, 2]
                        main_event_temp = events[index, 2]
                    elif events[index, 2] <= 10 and events[index, 2] % 2 == 1:
                        stim_chan[0, events[index, 0]] = self._ALPHA_CODE[
                            self.events_key_map[main_event_temp]
                        ][int(events[index, 2]/2)]
                    else:
                        continue
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
