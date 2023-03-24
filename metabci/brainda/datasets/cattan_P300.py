# -*- coding: utf-8 -*-
#
# Authors: Jieyu Wu <jieyu1999@tju.edu.cn>
# Date: 2023/2/9
# License: MIT License

"""
P300 datasets

"""


import numpy as np
from typing import Union, Optional, Dict
from pathlib import Path
from mne.channels import make_standard_montage
from mne import create_info
from mne.io import RawArray
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
from metabci.brainda.utils.channels import upper_ch_names
from metabci.brainda.utils.download import mne_data_path
import scipy.io as sci
# The filepath will be available when the dataset is uploaded
Cattan_P300_URL = "https://zenodo.org/record/2605205/files/"


class Cattan_P300(BaseTimeEncodingDataset):
    """
    Dataset in:
    Grégoire Cattan, Anton Andreev, Pedro Luiz Coelho Rodrigues
    and Marco Congedo,
    "Dataset of an EEG-based BCI experiment in Virtual Reality
    and on a Personal Computer,"
    in arXiv.1903.11297.1903.11297, 2019,

    This dataset contains electroencephalographic recordings on
    21 subjects doing a visual P300 experiment on PC
    (personal computer) and VR (virtual reality). The visual P300
    is an event-related potential elicited by a visual stimulation,
    peaking 240-600 ms after stimulus onset. The experiment was
    designed in order to compare the use of a P300-based
    brain-computer interface on a PC and with a virtual reality
    headset, concerning the physiological, subjective and
    performance aspects. The brain-computer interface is based
    on electroencephalography (EEG). EEG data were recorded thanks
    to 16 electrodes. The virtual reality headset consisted of a
    passive head-mounted display, that is, a head-mounted display
    which does not include any electronics with the exception of a
    smartphone. A full description of the experiment is available
    at https://hal.archives-ouvertes.fr/hal-02078533. This
    experiment was carried out at GIPSA-lab (University of Grenoble
    Alpes, CNRS, Grenoble-INP) in 2018, and promoted by the IHMTEK
    Company (Interaction Homme-Machine Technologie). The study was
    approved by the Ethical Committee of the University of Grenoble
    Alpes (Comité d’Ethique pour la Recherche Non-Interventionnelle).
    The ID of this dataset is VR.EEG.2018-GIPSA.


    This study implemented a P300 interface. Thirty-six characters
    were arranged as a 6 × 6 matrix displayed on the screen. The
    task of the subject was to focus on one of the characters.The
    experiment was composed of two sessions. One session ran under
    the PC condition and the other under the VR condition. The order
    of the session was randomized for all subjects. Each session
    comprised 12 blocks of five repetitions. All the repetitions
    within a block have the same target. A repetition consisted
    of 12 flashes of groups of six symbols chosen in such a way that
    after each repetition each symbol has flashed exactly two times.
    Thus, in each repetition the target symbol flashes twice, whereas
    the remaining ten flashes do not concern the target (non-target).
    The EEG signal was tagged corresponding to each flash.

    The recorded signals were bandpass-filtered at 0.1–100 Hz,
    notch-filtered at 50 Hz, digitized at a rate of 500 Hz and
    then stored in a computer. In this script, data is downscaled
    to 100 Hz.

    """

    _MINOR_TIME = 0.6
    _MINOR_EVENTS = {
        "1": (1, (0.0, _MINOR_TIME)),
        "2": (2, (0.0, _MINOR_TIME)),
        "3": (3, (0.0, _MINOR_TIME)),
        "4": (4, (0.0, _MINOR_TIME)),
        "5": (5, (0.0, _MINOR_TIME)),
        "6": (6, (0.0, _MINOR_TIME)),
        "7": (7, (0.0, _MINOR_TIME)),
        "8": (8, (0.0, _MINOR_TIME)),
        "9": (9, (0.0, _MINOR_TIME)),
        "10": (10, (0.0, _MINOR_TIME)),
        "11": (11, (0.0, _MINOR_TIME)),
        "12": (12, (0.0, _MINOR_TIME)),
    }

    _EVENTS = {
        "A": (211, (0, 35)),
        "B": (212, (0, 35)),
        "C": (213, (0, 35)),
        "D": (214, (0, 35)),
        "E": (215, (0, 35)),
        "F": (216, (0, 35)),
        "G": (221, (0, 35)),
        "H": (222, (0, 35)),
        "I": (223, (0, 35)),
        "J": (224, (0, 35)),
        "K": (225, (0, 35)),
        "L": (226, (0, 35)),
        "M": (231, (0, 35)),
        "N": (232, (0, 35)),
        "O": (233, (0, 35)),
        "P": (234, (0, 35)),
        "Q": (235, (0, 35)),
        "R": (236, (0, 35)),
        "S": (241, (0, 35)),
        "T": (242, (0, 35)),
        "U": (243, (0, 35)),
        "V": (244, (0, 35)),
        "W": (245, (0, 35)),
        "X": (246, (0, 35)),
        "Y": (251, (0, 35)),
        "Z": (252, (0, 35)),
        "1": (253, (0, 35)),
        "2": (254, (0, 35)),
        "3": (255, (0, 35)),
        "4": (256, (0, 35)),
        "5": (261, (0, 35)),
        "6": (262, (0, 35)),
        "7": (263, (0, 35)),
        "8": (264, (0, 35)),
        "9": (265, (0, 35)),
        "0": (266, (0, 35)),

    }

    _ALPHA_CODE = {
        "A": [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "B": [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "C": [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        "D": [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "E": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
        "F": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        "G": [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "H": [1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "I": [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        "J": [1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "K": [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
        "L": [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        "M": [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "N": [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "O": [1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        "p": [1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "Q": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1],
        "R": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        "S": [1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
        "T": [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
        "U": [1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1],
        "V": [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
        "W": [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        "X": [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
        "Y": [1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1],
        "Z": [1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1],
        "1": [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
        "2": [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1],
        "3": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1],
        "4": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
        "5": [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
        "6": [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1],
        "7": [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
        "8": [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],
        "9": [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
        "0": [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
    }

    _ENCODE_LOOP = 5

    _CHANNELS = ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz',
                 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    code_len = 12

    def __init__(self, paradigm='p300'):
        super().__init__(
            dataset_code="Cattan_P300",
            subjects=list(range(1, 13)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=500,
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
        if isinstance(subject, int):
            if subject < 10:
                P300_url = "{:s}subject_0{:d}_PC.mat".format(
                    Cattan_P300_URL, subject)
            else:
                P300_url = "{:s}subject_{:d}_PC.mat".format(
                    Cattan_P300_URL, subject)
        else:
            P300_url = "{:s}subject_0{:s}_PC.mat".format(
                Cattan_P300_URL, subject)
        dests = [
            [
                mne_data_path(
                    P300_url,
                    self.dataset_code,
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ]
        ]

        return dests

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = False,
            sess=None):
        sess = dict()
        runs = dict()
        dests = self.data_path(subject, update_path=True)
        dest = dests[0][0]
        raw_mat = sci.loadmat(dest)
        S = raw_mat['data']
        data = S[:, 1:17]
        ori_label = S[:, 17]
        row_non_target_label_loc = np.where((ori_label < 30) & (ori_label > 19))
        ori_label[row_non_target_label_loc] = ori_label[row_non_target_label_loc]-20+1
        col_non_target_label_loc = np.where((ori_label < 50) & (ori_label > 39))
        ori_label[col_non_target_label_loc] = ori_label[col_non_target_label_loc] - 40 + 1 + 6
        row_target_label_loc = np.where((ori_label < 70) & (ori_label > 59))
        ori_label[row_target_label_loc] = ori_label[row_target_label_loc] - 60 + 1
        col_target_label_loc = np.where((ori_label < 90) & (ori_label > 79))
        ori_label[col_target_label_loc] = ori_label[col_target_label_loc] - 80 + 1 + 6
        event_list = ori_label[ori_label != 0]
        big_label_loc = np.where(ori_label == 100)[0]
        target_mark = 2 * S[:, 18] + 1 * S[:, 19]
        target_mark = target_mark[target_mark != 0]
        value_list = list(self._ALPHA_CODE.values())
        big_label_list = list(self._EVENTS.values())
        for char_i in range(12):
            char_label_loc = event_list[char_i*66+1:char_i*66+13]
            char_target_mark = target_mark[char_i*60:char_i*60+12]
            tar_id = char_label_loc[np.where(char_target_mark == 2)]
            code = np.ones_like(char_label_loc, dtype=int)
            for tar_i in tar_id:
                code[int(tar_i-1)] = 2
            char_id = value_list.index(list(code))
            big_event = big_label_list[char_id][0]
            ori_label[big_label_loc[char_i]] = big_event
        ori_label[np.where(ori_label == 102)] = 0
        raw_events = np.zeros((np.shape(data)[0], 1))
        raw_events[:, 0] = ori_label

        data = np.append(1e-6 * data, raw_events, axis=1)
        ch_names = self._CHANNELS + ["stim"]
        ch_types = ["eeg"] * len(self._CHANNELS) + ["stim"]
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)
        raw = RawArray(data=data.T, info=info)
        raw = upper_ch_names(raw)
        montage = make_standard_montage('standard_1005')
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        raw.set_montage(montage)
        raw.resample(100)
        runs['run_1'] = raw
        if isinstance(subject, int):
            subject = str(subject)
        sess['subject_{:s}'.format(subject)] = runs
        return sess
