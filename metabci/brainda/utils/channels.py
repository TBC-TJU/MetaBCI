# -*- coding: utf-8 -*-
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/07
# License: MIT License
import mne
from mne.io import Raw
from typing import List, Union, Optional

def upper_ch_names(raw: Raw) -> Raw:
    """Uppercase all channel names in MNE Raw object.

    Parameters
    ----------
    raw : Raw
        MNE Raw object.

    Returns
    -------
    Raw
        MNE Raw object.
    """    
    # raw.info['ch_names'] = [ch_name.upper() for ch_name in raw.info['ch_names']]
    # for i, ch in enumerate(raw.info['chs']):
    #     ch['ch_name'] = raw.info['ch_names'][i]
    raw = raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.info['ch_names']})
    return raw

def pick_channels(ch_names: List[str], pick_chs: List[str], 
        ordered: bool = True, 
        match_case: str = 'auto') -> List[int]:
    """Wrapper of mne.pick_channels with match_case option.

    Parameters
    ----------
    ch_names : List[str]
        all channel names
    pick_chs : List[str]
        channel names to pick
    ordered : bool, optional
        if Ture, return picked channels in pick_chs order, by default True
    match_case : str, optional
        if True, pick channels in strict mode, by default 'auto'

    Returns
    -------
    List[int]
        indices of picked channels
    """    

    """Wrapper of mne.pick_channels with match_case option.
    """
    if match_case == 'auto':
        if len(set([ch_name.lower() for ch_name in ch_names])) < len(set(ch_names)):
            match_case = True
        else:
            match_case = False

    if match_case:
        picks = mne.pick_channels(ch_names, pick_chs, ordered=ordered)
    else:
        ch_names = [ch_name.lower() for ch_name in ch_names]
        pick_chs = [pick_ch.lower() for pick_ch in pick_chs]
        picks = mne.pick_channels(ch_names, pick_chs, ordered=ordered)

    return picks