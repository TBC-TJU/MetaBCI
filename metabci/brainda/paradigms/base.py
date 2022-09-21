# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/6/01
# License: MIT License
"""
Base Paradigm Design.

"""
from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
from mne.utils import verbose
from joblib import Parallel, delayed

from ..utils import pick_channels
from ..datasets.base import BaseDataset

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class BaseParadigm(metaclass=ABCMeta):
    """Abstract Base Paradigm.
    """

    def __init__(self, 
        channels: Optional[List[str]] = None, 
        events: Optional[List[str]] = None, 
        intervals: Optional[List[Tuple[float, float]]] = None, 
        srate: Optional[float] = None):
        """

        Parameters
        ----------
        channels : Optional[List[str]], optional
            selected channel names, if None use all channels in dataset, by default None
        events : Optional[List[str]], optional
            selected event names, if None use all events in dataset, by default None
        intervals : Optional[List[Tuple[Union[int, float]]]], optional
            selected event intervals, if None use default intervals in dataset.
            If only one interval passed, all events use the same interval.
            Otherwise the number of tuples should be the same as the number of events, by default None
        srate : Optional[float], optional
            sampling rate, if None use default srate in dataset, by default None
        """
        self.select_channels = None if channels is None else [ch_name.upper() for ch_name in channels]
        self.event_list = events
        self.intervals = intervals
        self.srate = srate
        self._raw_hook = None
        self._epochs_hook = None
        self._data_hook = None

    @abstractmethod
    def is_valid(self, dataset: BaseDataset) -> bool:
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : BaseDataset
            dataset
        """        
        pass
    
    def _map_events_intervals(self, dataset: BaseDataset):
        event_list = self.event_list
        intervals = self.intervals

        if event_list is None:
            # use all events in dataset
            event_list = list(dataset.events.keys())
        
        used_events = {ev: dataset.events[ev][0] for ev in event_list}

        if intervals is None:
            used_intervals = {ev: dataset.events[ev][1] for ev in event_list}
        elif len(intervals) == 1:
            used_intervals = {ev: intervals[0] for ev in event_list}
        else:
            if len(event_list) != len(intervals):
                raise ValueError("intervals should be the same number of events")
            used_intervals = {ev: interval for ev, interval in zip(event_list, intervals)}

        return used_events, used_intervals

    def register_raw_hook(self, hook):
        """Register raw hook before epoch operation.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process Raw object before epoch operation.
            Its' signature should look like:

            hook(raw, caches) -> raw, caches

            where caches is an dict stroing information, raw is MNE Raw instance.
        """
        self._raw_hook = hook

    def register_epochs_hook(self, hook):
        """Register epochs hook after epoch operation.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process Epochs object after epoch operation.
            Its' signature should look like:

            hook(epochs, caches) -> epochs, caches

            where caches is an dict storing information, epochs is MNE Epochs instance.
        """
        self._epochs_hook = hook

    def register_data_hook(self, hook):
        """Register data hook before return data.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process ndarray data before return it.
            Its' signature should look like:

            hook(X, y, meta, caches) -> X, y, meta, caches

            where caches is an dict storing information, X, y are ndarray object, meta is a pandas DataFrame instance.
        """
        self._data_hook = hook

    def unregister_raw_hook(self):
        """Unregister raw hook before epoch operation.
        
        """
        self._raw_hook = None

    def unregister_epochs_hook(self):
        """Register epochs hook after epoch operation.
        
        """
        self._epochs_hook = None

    def unregister_data_hook(self):
        """Register data hook before return data.
        
        """
        self._data_hook = None

    @verbose
    def _get_single_subject_data(self, dataset, subject_id, verbose=False):
        """Return data in micro-volt.
        """
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm. Check your events and channels settings".format(dataset.dataset_code))

        # # events, interval checking
        used_events, used_intervals = self._map_events_intervals(dataset)

        Xs = {}
        ys = {}
        metas = {}

        data = dataset.get_data([subject_id])

        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    # do raw hook either self-implemented or dataset inherited
                    caches = {}
                    if self._raw_hook:
                        raw, caches = self._raw_hook(raw, caches)
                    elif hasattr(dataset, 'raw_hook'):
                        raw, caches = dataset.raw_hook(raw, caches)

                    # pick selected channels by order
                    channels = dataset.channels if self.select_channels is None else self.select_channels
                    picks = pick_channels(raw.ch_names, channels, ordered=True)

                    # find available events, first check stim_channels then annotations
                    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
                    if len(stim_channels) > 0:
                        events = mne.find_events(raw, shortest_event=0, initial_event=True)
                    else:
                        # convert event_id to its number type instead of default auto-renaming in 0.19.2
                        events, _ = mne.events_from_annotations(raw, event_id=(lambda x: int(x)))

                    for event_name in used_events.keys():
                        # mne.pick_events returns any matching events in include
                        # only raise Runtime Error when nothing is found
                        # then we just skip this event
                        try:
                            selected_events = mne.pick_events(events, include=used_events[event_name])
                        except RuntimeError:
                            continue 

                        # transform Raw to Epochs

                        epochs = mne.Epochs(raw, selected_events,
                            event_id={event_name: used_events[event_name]},
                            event_repeated='drop', 
                            tmin=used_intervals[event_name][0],
                            tmax=used_intervals[event_name][1]-1./raw.info['sfreq'],
                            picks=picks,
                            proj=False, baseline=None, preload=True)
                        
                        # skip invalid time intervals
                        if len(epochs) == 0:
                            continue

                        # do epochs hook
                        if self._epochs_hook:
                            epochs, caches = self._epochs_hook(epochs, caches)
                        elif hasattr(dataset, 'epochs_hook'):
                            epochs, caches = dataset.epochs_hook(epochs, caches)
                        
                        # FIXME: is this resample reasonable?
                        if self.srate:
                            # as MNE suggested, decimate after extract epochs
                            # low-pass raw object in raw_hook to prevent aliasing problem 
                            epochs = epochs.resample(self.srate)
                            # epochs = epochs.decimate(dataset.srate//self.srate)

                        # retrieve X, y and meta
                        X = epochs.get_data() * 1e6 # micro-volt default
                        y = epochs.events[:, -1]
                        trial_ids = np.argwhere(events[:, -1] == list(epochs.event_id.values())[0]).reshape((-1))
                        meta = pd.DataFrame(
                            {
                                "subject": [subject]*len(epochs),
                                "session": [session]*len(epochs),
                                "run": [run]*len(epochs), 
                                "event": [event_name]*len(epochs),
                                "trial_id": trial_ids,
                                "dataset": [dataset.dataset_code]*len(epochs)
                            })

                        # do data hook
                        if self._data_hook:
                            X, y, meta, caches = self._data_hook(X, y, meta, caches)
                        elif hasattr(dataset, 'data_hook'):
                            X, y, meta, caches = dataset.data_hook(X, y, meta, caches)

                        # collecting data
                        pre_X = Xs.get(event_name)
                        if pre_X is not None:
                            Xs[event_name] = np.concatenate((pre_X, X), axis=0)
                        else:
                            Xs[event_name] = X

                        pre_y = ys.get(event_name)
                        if pre_y is not None:
                            ys[event_name] = np.concatenate((pre_y, y), axis=0)
                        else:
                            ys[event_name] = y

                        pre_meta = metas.get(event_name)
                        if pre_meta is not None:
                            metas[event_name] = pd.concat(
                                (pre_meta, meta),
                                axis=0,
                                ignore_index=True
                            )
                        else:
                            metas[event_name] = meta  
        return Xs, ys, metas

    @verbose
    def get_data(self, dataset: BaseDataset, 
            subjects: Optional[List[Union[int, str]]] = None, 
            label_encode: bool = True,
            return_concat: bool = False, 
            n_jobs: int = -1, 
            verbose: Optional[bool] = None) -> Tuple[Union[Dict[str, Union[np.ndarray, pd.DataFrame]], Union[np.ndarray, pd.DataFrame]], ...]:
        """Get data from dataset with selected subjects.

        Parameters
        ----------
        dataset : BaseDataset
            dataset
        subjects : Optional[List[Union[int, str]]], optional
            selected subjects, by default None
        label_encode: bool, optional,
            if True, return y in label encode way
        return_concat : bool, optional
            if True, return concated ndarray object, otherwise return dict of events, by default False
        n_jobs : int, optional
            Parallel jobs, by default -1
        verbose : Optional[bool], optional
            verbose, by default None

        Returns
        -------
        Tuple[Union[Dict[str, Union[np.ndarray, pd.DataFrame]], Union[np.ndarray, pd.DataFrame]], ...]
            Xs, ys, metas, corresponding to data, label and meta data

        Raises
        ------
        TypeError
            raise error if dataset is not avaliable for the paradigm
        """
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm. Check your events and channels settings".format(
                    dataset.dataset_code))
        # events, interval checking
        used_events, used_intervals = self._map_events_intervals(dataset)

        Xs = {}
        ys = {}
        metas = {}

        X, y, meta = zip(*Parallel(n_jobs=n_jobs)(delayed(self._get_single_subject_data)(dataset, sub_id, verbose=verbose) for sub_id in subjects))

        for event_name in used_events.keys():
            Xs[event_name] = np.concatenate([X[i][event_name] for i in range(len(subjects)) if event_name in X[i]], axis=0)
            ys[event_name] =  np.concatenate([y[i][event_name] for i in range(len(subjects)) if event_name in y[i]], axis=0)
            metas[event_name] = pd.concat([meta[i][event_name] for i in range(len(subjects)) if event_name in meta[i]], axis=0, ignore_index=True)

        if label_encode:
            event_list = list(used_events.keys())
            event_id = [dataset.events[e][0] for e in event_list]
            for event_name in used_events.keys():
                ys[event_name] = label_encoder(ys[event_name], event_id)

        # python gaurante values in insert order.
        if return_concat:
            Xs = np.concatenate(list(Xs.values()), axis=0)
            ys = np.concatenate(list(ys.values()), axis=0)
            metas = pd.concat(list(metas.values()), axis=0, ignore_index=True)

        return Xs, ys, metas

    def __str__(self):
        desc = "{}".format(self.__class__.__name__)
        return desc

