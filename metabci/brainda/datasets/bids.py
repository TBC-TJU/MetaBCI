# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023-10-4
# License: MIT License
import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from mne.channels import make_standard_montage
from mne.io import Raw
from mne_bids import (BIDSPath, get_entity_vals, read_raw_bids)
from ..utils.download import mne_data_path
from .base import BaseDataset

warnings.filterwarnings("ignore")

BASE_URL = "https://osf.io/download/8rbfk?version=1"


class matchingpennies(BaseDataset):
    """An example BIDS format dataset.
    This dataset is an standard example of a BIDS format dataset, that
    mentioned in [1], and now it can be downloaded from [2]. However, as
    the suggestion in [3], we download the dataset from BASE_URL instead.
    The source reference of this dataset is [4].

    This is the "Matching Pennies" dataset. It was collected as part of
    a small scale replication project targeting the following reference [5]

    In brief, it contains EEG data for 7 subjects raising either their left
    or right hand, thus giving rise to a lateralized readiness potential as
    measured with the EEG. For details, see the Details about the experiment
    section.

    References:
    [1] Pernet, C.R., Appelhoff, S., Gorgolewski, K.J. et al.
        EEG-BIDS, an extension to the brain imaging data structure for
        electroencephalography. Sci Data 6, 103 (2019).
        https://doi.org/10.1038/s41597-019-0104-8
    [2] https://gin.g-node.org/sappelhoff/eeg_matchingpennies
    [3] https://github.com/mne-tools/mne-bids-pipeline/blob/main/mne_bids_pipeline/tests/datasets.py
    [4] Appelhoff, S., Sauer, D. & Gill, S. S. Matching Pennies:
        A Brain Computer Interface Implementation Dataset.
        Open Science Framework, https://doi.org/10.17605/OSF.IO/CJ2DR (2018).
    [5] Matthias Schultze-Kraft et al. "Predicting Motor Intentions with
        Closed-Loop Brain-Computer Interfaces". In: Springer Briefs in
        Electrical and Computer Engineering. Springer International Publishing,
        2017, pp. 79~90.
    """
    _EVENTS = {
        "left": (1, (0, 3)),
        "right": (2, (0, 3)),
    }
    _CHANNELS = [
        'FC5', 'FC1', 'C3', 'CP5', 'CP1', 'FC2', 'FC6', 'C4', 'CP2', 'CP6'
    ]

    def __init__(self):
        super().__init__(
            dataset_code='matchingpennies',
            subjects=list(range(1, 8)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm='movement_intention',
        )
        self.data_dest = mne_data_path(
            BASE_URL,
            sign=self.dataset_code,
            path=None,
            force_update=False,
            update_path=None,
            proxies=None,
            verbose=None
        )
        # check if the data_dest is a folder
        if not os.path.isdir(self.data_dest):
            # modify the file name to add the .zip extension
            zip_name = self.data_dest + '.zip'
            # rename the file
            os.rename(self.data_dest, zip_name)
            # add the .zip extension to the file name
            self.data_dest = zip_name
            # unzip the file
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(self.data_dest[:-4])
            # get the upzip folder name
            unzip_folder = os.listdir(self.data_dest[:-4])[0]
            # add the upzip folder name to the data_dest\
            self.data_dest = os.path.join(self.data_dest[:-4], unzip_folder)
        else:
            self.data_dest += '/eeg_matchingpennies'
        self.dataset_subjects = get_entity_vals(self.data_dest, 'subject')

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
            raise ValueError(
                f"Invalid subject id {subject}. "
                f"Valid ids are {self.subjects}"
            )

        bids_path = BIDSPath(
            root=self.data_dest,
            datatype='eeg'
        )

        dests = []

        dests = [
            [
                bids_path.update(
                    subject=self.dataset_subjects[int(subject)-1],
                    task='matchingpennies')
            ]
        ]

        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int],
        verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_path in enumerate(run_dests):
                raw = read_raw_bids(
                    run_path,
                    extra_params=dict(preload=True),
                    verbose=verbose)
                raw.set_montage(montage)
                raw.rename_channels(
                    {ch_name: ch_name.upper() for ch_name in raw.ch_names}
                )
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess
