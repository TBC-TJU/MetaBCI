# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/01/07
# License: MIT License
"""
Zhou2016.
"""
import os
import zipfile
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

from mne.io import read_raw_cnt, Raw
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names

ZHOU_URL = "https://ndownloader.figshare.com/files/3662952"


class Zhou2016(BaseDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """

    _EVENTS = {"left_hand": (1, (0, 5)), "right_hand": (2, (0, 5)), "feet": (3, (0, 5))}

    _CHANNELS = [
        "FP1",
        "FP2",
        "FC3",
        "FCZ",
        "FC4",
        "C3",
        "CZ",
        "C4",
        "CP3",
        "CPZ",
        "CP4",
        "O1",
        "OZ",
        "O2",
    ]

    def __init__(self):
        super().__init__(
            dataset_code="zhou2016",
            subjects=list(range(1, 5)),
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
        url = "{:s}".format(ZHOU_URL)
        file_dest = mne_data_path(
            url,
            self.dataset_code,
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )
        parent_dir = Path(file_dest).parent

        if not os.path.exists(os.path.join(parent_dir, "data")):
            # decompression the data
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=parent_dir)
        dests: List[List[Union[str, Path]]] = []
        for session in range(1, 4):
            runs: List[Union[str, Path]] = []
            for run in ["A", "B"]:
                runs.append(
                    os.path.join(
                        parent_dir,
                        "data",
                        "S{:d}_{:d}{:s}.cnt".format(subject, session, run),
                    )
                )
            dests.append(runs)
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
                raw = read_raw_cnt(run_file, eog=["VEOU", "VEOL"], preload=True)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)

                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess
