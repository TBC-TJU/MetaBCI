# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/27
# License: MIT License
"""
Alex Motor imagery dataset.
"""
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

from mne.io import Raw, read_raw_fif
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names

ALEX_URL = 'https://zenodo.org/record/806023/files/'


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset.

    Motor imagery dataset from the PhD dissertation of A. Barachant [1]_.

    This Dataset contains EEG recordings from 8 subjects, performing 2 task of
    motor imagination (right hand, feet or rest). Data have been recorded at
    512Hz with 16 wet electrodes (Fpz, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
    P7, P3, Pz, P4, P8) with a g.tec g.USBamp EEG amplifier.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a trial is encoded as 1,
    then the actual start of the motor imagination is encoded with 2 for
    imagination of a right hand movement, 3 for imagination of both feet
    movement and 4 with a rest trial.

    The duration of each trial is 3 second. There is 20 trial of each class.

    references
    ----------
    .. [1] Barachant, A., 2012. Commande robuste d'un effecteur par une
           interface cerveau machine EEG asynchrone (Doctoral dissertation,
           Université de Grenoble).
           https://tel.archives-ouvertes.fr/tel-01196752

    """
    
    _EVENTS = {
        "right_hand": (2, (0, 3)),
        "feet": (3, (0, 3)),
        "rest": (4, (0, 3))
    }

    _CHANNELS = [
        'FPZ','F7','F3','FZ','F4','F8',
        'T7','C3','C4','T8',
        'P7','P3','PZ','P4','P8'
    ]
    
    def __init__(self):
        super().__init__(
            dataset_code='alexeeg', 
            subjects=list(range(1, 9)),
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
        url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject)
        dests = [
            [mne_data_path(url, self.dataset_code, 
                path=path, proxies=proxies, force_update=force_update, update_path=update_path)]
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
                raw = Raw(run_file, preload=True)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs

        return sess

        



