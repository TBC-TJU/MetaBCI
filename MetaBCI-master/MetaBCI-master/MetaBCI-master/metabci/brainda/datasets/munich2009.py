# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/3/4
# License: MIT License
"""
Munich MI dataset.
Unkown channel names.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

from mne.io import Raw, read_raw_eeglab
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names

MUNICH_URL = "https://zenodo.org/record/1217449/files/"


class MunichMI(BaseDataset):
    """Munich Motor Imagery dataset.

    Motor imagery dataset from Grosse-Wentrup et al. 2009 [1]_.

    A trial started with the central display of a white fixation cross. After 3
    s, a white arrow was superimposed on the fixation cross, either pointing to
    the left or the right.
    Subjects were instructed to perform haptic motor imagery of the
    left or the right hand during display of the arrow, as indicated by the
    direction of the arrow. After another 7 s, the arrow was removed,
    indicating the end of the trial and start of the next trial. While subjects
    were explicitly instructed to perform haptic motor imagery with the
    specified hand, i.e., to imagine feeling instead of visualizing how their
    hands moved, the exact choice of which type of imaginary movement, i.e.,
    moving the fingers up and down, gripping an object, etc., was left
    unspecified.
    A total of 150 trials per condition were carried out by each subject,
    with trials presented in pseudorandomized order.

    Ten healthy subjects (S1–S10) participated in the experimental
    evaluation. Of these, two were females, eight were right handed, and their
    average age was 25.6 years with a standard deviation of 2.5 years. Subject
    S3 had already participated twice in a BCI experiment, while all other
    subjects were naive to BCIs. EEG was recorded at M=128 electrodes placed
    according to the extended 10–20 system. Data were recorded at 500 Hz with
    electrode Cz as reference. Four BrainAmp amplifiers were used for this
    purpose, using a temporal analog high-pass filter with a time constant of
    10 s. The data were re-referenced to common average reference
    offline. Electrode impedances were below 10 kΩ for all electrodes and
    subjects at the beginning of each recording session. No trials were
    rejected and no artifact correction was performed. For each subject, the
    locations of the 128 electrodes were measured in three dimensions using a
    Zebris ultrasound tracking system and stored for further offline analysis.


    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, et al. "Beamforming in noninvasive
           brain–computer interfaces." IEEE Transactions on Biomedical
           Engineering 56.4 (2009): 1209-1219.

    """

    _EVENTS = {
        "left_hand": (10, (0, 7)),
        "right_hand": (20, (0, 7)),
    }

    _CHANNELS = [str(i) for i in range(1, 129)]

    def __init__(self):
        super().__init__(
            dataset_code="munichmi",
            subjects=list(range(1, 11)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=500,
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
        url = "{:s}subject{:d}.fdt".format(MUNICH_URL, subject)
        mne_data_path(
            url,
            self.dataset_code,
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )

        url = "{:s}subject{:d}.set".format(MUNICH_URL, subject)
        dests = [
            [
                mne_data_path(
                    url,
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
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)

        # DON'T KNOW CHANNEL NAMES!!!
        # montage = make_standard_montage('standard_1005')
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_array in enumerate(run_dests):
                raw = read_raw_eeglab(run_array, preload=True)
                raw = upper_ch_names(raw)
                # raw.set_montage(montage)
                raw.annotations.delete(0)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess
