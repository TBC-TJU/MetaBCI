# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/01/07
# License: MIT License
"""
Nakanishi SSVEP dataset.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

import numpy as np
from mne import create_info
from mne.io import RawArray, Raw
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

Nakanishi2015_URL = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"


class Nakanishi2015(BaseDataset):
    """SSVEP Nakanishi 2015 dataset

    This dataset contains 12-class joint frequency-phase modulated steady-state
    visual evoked potentials (SSVEPs) acquired from 10 subjects used to
    estimate an online performance of brain-computer interface (BCI) in the
    reference study [1]_.

    references
    ----------
    .. [1] Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
    "A Comparison Study of Canonical Correlation Analysis Based Methods for
    Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10,
    e140703, 2015.
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703
    """

    _CHANNELS = ["PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2"]

    _FREQS = [
        9.25,
        11.25,
        13.25,
        9.75,
        11.75,
        13.75,
        10.25,
        12.25,
        14.25,
        10.75,
        12.75,
        14.75,
    ]
    _PHASES = [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]

    _EVENTS = {str(freq): (i + 1, (0, 4)) for i, freq in enumerate(_FREQS)}

    def __init__(self):
        super().__init__(
            dataset_code="nakanishi2015",
            subjects=list(range(1, 11)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=256,
            paradigm="ssvep",
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
        url = "{:s}s{:d}.mat".format(Nakanishi2015_URL, subject)
        file_dest = mne_data_path(
            url,
            self.dataset_code,
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )

        dests = [[file_dest]]
        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        dests = self.data_path(subject)
        raw_mat = loadmat(dests[0][0])
        n_samples, n_channels, n_trials = 1114, 8, 15
        n_classes = 12

        data = np.transpose(raw_mat["eeg"], axes=(0, 3, 1, 2))
        data = np.reshape(data, newshape=(-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 38] = np.array(
            [n_trials * [i + 1] for i in range(n_classes)]
        ).flatten()
        data = np.concatenate([1e-6 * data, raw_events], axis=1)

        buff = (data.shape[0], n_channels + 1, 50)
        data = np.concatenate([np.zeros(buff), data, np.zeros(buff)], axis=2)
        ch_names = self._CHANNELS + ["stim"]
        ch_types = ["eeg"] * len(self._CHANNELS) + ["stim"]

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info)
        raw = upper_ch_names(raw)
        raw.set_montage(montage)

        sess = {"session_0": {"run_0": raw}}
        return sess

    def get_freq(self, event: str):
        return self._FREQS[self._EVENTS[event][0] - 1]

    def get_phase(self, event: str):
        return self._PHASES[self._EVENTS[event][0] - 1]
