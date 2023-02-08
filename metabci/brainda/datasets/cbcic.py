# -*- coding: utf-8 -*-
"""
China BCI Competition.
"""
import mne

from metabci.brainda.utils.download import mne_data_path
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

import numpy as np
from mne import create_info
from mne.io import Raw, RawArray, read_raw_edf
from mne.channels import make_standard_montage
from .base import BaseDataset, BaseTimeEncodingDataset
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

# no available links now
CBCIC2019001_URL = "file:///CBCIC2019001"
CBCIC2019004_URL = "file:///CBCIC2019004"
CBCIC2020aVEP_URL = "file://CBCIC2020aVEP"


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
        "FPZ",
        "FP1",
        "FP2",
        "AF3",
        "AF4",
        "AF7",
        "AF8",
        "FZ",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "FCZ",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "FC5",
        "FC6",
        "FT7",
        "FT8",
        "CZ",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "T7",
        "T8",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "CP5",
        "CP6",
        "TP7",
        "TP8",
        "PZ",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "POZ",
        "PO3",
        "PO4",
        "PO5",
        "PO6",
        "PO7",
        "PO8",
        "OZ",
        "O1",
        "O2",
    ]

    def __init__(self):
        super().__init__(
            dataset_code="cbcic2019001",
            subjects=list(range(1, 19)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm="imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        subject = cast(int, subject)
        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        if subject in [6, 14, 15, 18]:
            file_name = "T{:02d}01T.mat".format(subject)
        else:
            file_name = "B{:02d}01T.mat".format(subject)

        url = "{:s}/{:02d}/{:s}".format(CBCIC2019001_URL, subject, file_name)
        dests = [
            [
                mne_data_path(
                    url,
                    "cbcic",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ]
        ]
        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw_mat = loadmat(run_file)["EEG"]
                epoch_data = raw_mat["data"][:-5] * 1e-6
                stim = np.zeros((1, epoch_data.shape[-1]))
                for event in raw_mat["event"]:
                    stim[0, int(event["latency"]) - 1] = int(event["type"])
                data = np.concatenate((epoch_data, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in self._CHANNELS]
                ch_types = ["eeg"] * len(ch_names)
                ch_names = ch_names + ["STI 014"]
                ch_types = ch_types + ["stim"]

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate
                )

                raw = RawArray(data=data, info=info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
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
        "FPZ",
        "FP1",
        "FP2",
        "AF3",
        "AF4",
        "AF7",
        "AF8",
        "FZ",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "FCZ",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "FC5",
        "FC6",
        "FT7",
        "FT8",
        "CZ",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "T7",
        "T8",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "CP5",
        "CP6",
        "TP7",
        "TP8",
        "PZ",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "POZ",
        "PO3",
        "PO4",
        "PO5",
        "PO6",
        "PO7",
        "PO8",
        "OZ",
        "O1",
        "O2",
    ]

    def __init__(self):
        super().__init__(
            dataset_code="cbcic2019004",
            subjects=list(range(1, 7)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:

        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        subject = cast(int, subject)
        runs = []
        for i in range(1, 5):
            url = "{:s}/{:02d}/block{:d}.mat".format(
                CBCIC2019004_URL, subject, i)
            runs.append(
                mne_data_path(
                    url,
                    "cbcic",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            )

        dests = [runs]
        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw_mat = loadmat(run_file)
                epoch_data = raw_mat["data"][:-6] * 1e-6
                stims = raw_mat["data"][-1][np.newaxis, :]
                data = np.concatenate((epoch_data, stims), axis=0)

                ch_names = [ch_name.upper() for ch_name in self._CHANNELS]
                ch_types = ["eeg"] * len(ch_names)
                ch_names = ch_names + ["STI 014"]
                ch_types = ch_types + ["stim"]

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate
                )

                raw = RawArray(data=data, info=info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess


class XuaVEPDataset(BaseTimeEncodingDataset):
    """

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
            raise ValueError('Invalid subject {} given'.format(subject))

        runs = list(range(1, 7))
        sessions = list(range(1))
        base_url = CBCIC2020aVEP_URL
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
                data_path = '{:s}/Sub{:s}/session_0{:s}.edf'.format(
                    base_url, sub_name, str(run))
                event_path = '{:s}/Sub{:s}/session_0{:s}_events.edf'.format(
                    base_url, sub_name, str(run))
                dests.append((data_path, event_path))
            sessions_dests.append(dests)
        return sessions_dests

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = False
    ):
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        # montage = mne.channels.read_custom_montage(os.path.join(filepath, '64-channels.loc'))
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
