# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/28
# License: MIT License
"""
TUNERL Datasets

Weibo2014
"""
import os, zipfile
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import mne
from mne import create_info
from mne.io import Raw, read_raw_cnt, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

Weibo2014_URLs = [
    'https://dataverse.harvard.edu/api/access/datafile/2499178',
    'https://dataverse.harvard.edu/api/access/datafile/2499182',
    'https://dataverse.harvard.edu/api/access/datafile/2499179']


class Weibo2014(BaseDataset):
    """Motor Imagery dataset from Weibo et al 2014.

    Dataset from the article *Evaluation of EEG oscillatory patterns and
    cognitive process during simple and compound limb motor imagery* [1]_.

    It contains data recorded on 10 subjects, with 60 electrodes.

    This dataset was used to investigate the differences of the EEG patterns
    between simple limb motor imagery and compound limb motor
    imagery. Seven kinds of mental tasks have been designed, involving three
    tasks of simple limb motor imagery (left hand, right hand, feet), three
    tasks of compound limb motor imagery combining hand with hand/foot
    (both hands, left hand combined with right foot, right hand combined with
    left foot) and rest state.

    At the beginning of each trial (8 seconds), a white circle appeared at the
    center of the monitor. After 2 seconds, a red circle (preparation cue)
    appeared for 1 second to remind the subjects of paying attention to the
    character indication next. Then red circle disappeared and character
    indication (‘Left Hand’, ‘Left Hand & Right Foot’, et al) was presented on
    the screen for 4 seconds, during which the participants were asked to
    perform kinesthetic motor imagery rather than a visual type of imagery
    while avoiding any muscle movement. After 7 seconds, ‘Rest’ was presented
    for 1 second before next trial (Fig. 1(a)). The experiments were divided
    into 9 sections, involving 8 sections consisting of 60 trials each for six
    kinds of MI tasks (10 trials for each MI task in one section) and one
    section consisting of 80 trials for rest state. The sequence of six MI
    tasks was randomized. Intersection break was about 5 to 10 minutes.

    References
    -----------
    .. [1] Yi, Weibo, et al. "Evaluation of EEG oscillatory patterns and
           cognitive process during simple and compound limb motor imagery."
           PloS one 9.12 (2014). https://doi.org/10.1371/journal.pone.0114853
    """

    _EVENTS = {
        "left_hand": (1, (3, 7)),
        "right_hand": (2, (3, 7)), 
        "hands": (3, (3, 7)), 
        "feet": (4, (3, 7)),
        "left_hand_right_foot": (5, (3, 7)), 
        "right_hand_left_foot": (6, (3, 7)), 
        "rest": (7, (3, 7))
    }

    _CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
        'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
        'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    
    def __init__(self):
        super().__init__(
            dataset_code="weibo2014",
            subjects=list(range(1, 11)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=200,
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

        if subject in range(1, 5):
            sub_names = ["cl", "cyy", "kyf", "lnn"]
            inc = 0
            file_dest = mne_data_path(Weibo2014_URLs[0], 'tunerl', 
                path=path, proxies=proxies, force_update=force_update, update_path=update_path)
        elif subject in range(5, 8):
            sub_names = ["ls", "ry", "wcf"]
            inc = 4
            file_dest = mne_data_path(Weibo2014_URLs[1], 'tunerl', 
                path=path, proxies=proxies, force_update=force_update, update_path=update_path)
        else:
            sub_names = ["wx", "yyx", "zd"]
            inc = 7
            file_dest = mne_data_path(Weibo2014_URLs[2], 'tunerl', 
                path=path, proxies=proxies, force_update=force_update, update_path=update_path)            
        
        parent_dir = Path(file_dest).parent

        if not os.path.exists(os.path.join(parent_dir, 'subject_{:d}.mat'.format(subject))):
            with zipfile.ZipFile(file_dest, 'r') as archive:
                archive.extractall(path=parent_dir)
            for i, sub_name in enumerate(sub_names):
                os.rename(
                    os.path.join(parent_dir, '{}.mat'.format(sub_name)),
                    os.path.join(parent_dir, 'subject_{:d}.mat'.format(i+inc+1))
                )

        dests = [
            [
                os.path.join(parent_dir, 'subject_{:d}.mat'.format(subject))
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
                raw_mat = loadmat(run_file)
                epoch_data = raw_mat['data'] * 1e-6
                label = raw_mat['label']

                stim = np.zeros((1, epoch_data.shape[1], epoch_data.shape[2]))
                stim[0, 0, :] = label

                data = np.concatenate((epoch_data, stim), axis=0)
                data = np.transpose(data, axes=(0, 2, 1))
                data = np.reshape(data, (data.shape[0], -1))

                ch_names = [ch_name.upper() for ch_name in  self._CHANNELS] + ['VEO', 'HEO', 'STI 014']
                ch_names.insert(57, 'CB1')
                ch_names.insert(61, 'CB2')
                ch_types = ['eeg']*62 + ['eog']*2
                ch_types[57] = 'misc'
                ch_types[61] = 'misc'
                ch_types = ch_types + ['stim']

                info = create_info(
                    ch_names=ch_names, ch_types=ch_types, sfreq=self.srate)
                
                raw = RawArray(data=data, info=info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
