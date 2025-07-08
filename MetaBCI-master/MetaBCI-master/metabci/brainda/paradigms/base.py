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
from ..datasets.base import BaseDataset, BaseTimeEncodingDataset


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = y == label
        new_y[ix] = i
    return new_y


class BaseParadigm(metaclass=ABCMeta):
    """Abstract Base Paradigm."""

    def __init__(
            self,
            channels: Optional[List[str]] = None,
            events: Optional[List[str]] = None,
            intervals: Optional[List[Tuple[float, float]]] = None,
            srate: Optional[float] = None,
    ):
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
        self.select_channels = (
            None if channels is None else [
                ch_name.upper() for ch_name in channels]
        )
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
        """Select and map events with their inervals.

        Args:
            dataset (BaseDataset): a pre defined dataset

        Raises:
            ValueError: length of intervals should be the same number of events

        Returns:
            used_evnets: selected events, return in dict.
            used_intervals: intervals of selected events, return in dict
        """
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
                raise ValueError(
                    "intervals should be the same number of events")
            used_intervals = {
                ev: interval for ev, interval in zip(event_list, intervals)
            }

        return used_events, used_intervals

    def register_raw_hook(self, hook):
        """Register raw hook before epoch operation.

        Parameters
        ----------
        hook : callable object
            Callable object to process Raw object before epoch operation.
            Its signature should look like:

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
        """Unregister raw hook before epoch operation."""
        self._raw_hook = None

    def unregister_epochs_hook(self):
        """Register epochs hook after epoch operation."""
        self._epochs_hook = None

    def unregister_data_hook(self):
        """Register data hook before return data."""
        self._data_hook = None

    @verbose
    def _get_single_subject_data(self, dataset, subject_id, verbose=False):
        """Return data in micro-volt."""
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm. Check your events and channels settings".format(
                    dataset.dataset_code
                )
            )

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
                    elif hasattr(dataset, "raw_hook"):
                        raw, caches = dataset.raw_hook(raw, caches)

                    # pick selected channels by order
                    channels = (
                        dataset.channels
                        if self.select_channels is None
                        else self.select_channels
                    )
                    picks = pick_channels(raw.ch_names, channels, ordered=True)

                    # find available events, first check stim_channels then annotations
                    stim_channels = mne.utils._get_stim_channel(
                        None, raw.info, raise_error=False
                    )
                    if len(stim_channels) > 0:
                        events = mne.find_events(
                            raw, shortest_event=0, initial_event=True
                        )
                    else:
                        # convert event_id to its number type instead of default auto-renaming in 0.19.2
                        events, _ = mne.events_from_annotations(
                            raw, event_id=(lambda x: int(x))
                        )

                    for event_name in used_events.keys():
                        # mne.pick_events returns any matching events in include
                        # only raise Runtime Error when nothing is found
                        # then we just skip this event
                        try:
                            selected_events = mne.pick_events(
                                events, include=used_events[event_name]
                            )
                        except RuntimeError:
                            continue

                        # transform Raw to Epochs

                        epochs = mne.Epochs(
                            raw,
                            selected_events,
                            event_id={event_name: used_events[event_name]},
                            event_repeated="drop",
                            tmin=used_intervals[event_name][0],
                            tmax=used_intervals[event_name][1] - 1.0 / raw.info["sfreq"],
                            picks=picks,
                            proj=False,
                            baseline=None,
                            preload=True,
                        )

                        # skip invalid time intervals
                        if len(epochs) == 0:
                            continue

                        # do epochs hook
                        if self._epochs_hook:
                            epochs, caches = self._epochs_hook(epochs, caches)
                        elif hasattr(dataset, "epochs_hook"):
                            epochs, caches = dataset.epochs_hook(
                                epochs, caches)

                        # FIXME: is this resample reasonable?
                        if self.srate:
                            # as MNE suggested, decimate after extract epochs
                            # low-pass raw object in raw_hook to prevent aliasing problem
                            epochs = epochs.resample(self.srate)
                            # epochs = epochs.decimate(dataset.srate//self.srate)

                        # retrieve X, y and meta
                        X = epochs.get_data() * 1e6  # micro-volt default
                        y = epochs.events[:, -1]
                        trial_ids = np.argwhere(
                            events[:, -1] == list(epochs.event_id.values())[0]
                        ).reshape((-1))
                        meta = pd.DataFrame(
                            {
                                "subject": [subject] * len(epochs),
                                "session": [session] * len(epochs),
                                "run": [run] * len(epochs),
                                "event": [event_name] * len(epochs),
                                "trial_id": trial_ids,
                                "dataset": [dataset.dataset_code] * len(epochs),
                            }
                        )

                        # do data hook
                        if self._data_hook:
                            X, y, meta, caches = self._data_hook(
                                X, y, meta, caches)
                        elif hasattr(dataset, "data_hook"):
                            X, y, meta, caches = dataset.data_hook(
                                X, y, meta, caches)

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
                                (pre_meta, meta), axis=0, ignore_index=True
                            )
                        else:
                            metas[event_name] = meta
        return Xs, ys, metas

    @verbose
    def get_data(
            self,
            dataset: BaseDataset,
            subjects: List[Union[int, str]] = [],
            label_encode: bool = True,
            return_concat: bool = False,
            n_jobs: int = -1,
            verbose: Optional[bool] = None,
    ) -> Tuple[
        Union[
            Dict[str, Union[np.ndarray, pd.DataFrame]],
            Union[np.ndarray, pd.DataFrame]
        ],
        ...,
    ]:
        """Get data from dataset with selected subjects.

        Parameters
        ----------
        dataset : BaseDataset
            dataset
        subjects : List[Union[int, str]],
            selected subjects, by default empty
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
                    dataset.dataset_code
                )
            )
        # events, interval checking
        used_events, used_intervals = self._map_events_intervals(dataset)

        Xs = {}
        ys = {}
        metas = {}

        X, y, meta = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(self._get_single_subject_data)(
                    dataset, sub_id, verbose=verbose)
                for sub_id in subjects
            )
        )

        for event_name in used_events.keys():
            Xs[event_name] = np.concatenate(
                [X[i][event_name]
                 for i in range(len(subjects)) if event_name in X[i]],
                axis=0,
            )
            ys[event_name] = np.concatenate(
                [y[i][event_name]
                 for i in range(len(subjects)) if event_name in y[i]],
                axis=0,
            )
            metas[event_name] = pd.concat(
                [
                    meta[i][event_name]
                    for i in range(len(subjects))
                    if event_name in meta[i]
                ],
                axis=0,
                ignore_index=True,
            )

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


class BaseTimeEncodingParadigm(BaseParadigm):

    def __init__(
            self,
            channels: Optional[List[str]] = None,
            events: Optional[List[str]] = None,
            intervals: Optional[List[Tuple[float, float]]] = None,
            minor_event_intervals: Optional[List[Tuple[float, float]]] = None,
            srate: Optional[float] = None,
    ):

        super().__init__(
            channels=channels,
            events=events,
            intervals=intervals,
            srate=srate
        )

        self._trial_hook = None
        self.minor_event_intervals = minor_event_intervals

    def is_valid(self, dataset):
        pass

    def _map_events_intervals(self, dataset):
        event_list = self.event_list
        intervals = self.intervals
        minor_event_intervals = self.minor_event_intervals

        if event_list is None:
            # If no given events, using the dataset defined events
            event_list = list(dataset.events.keys())

        used_events = {ev: dataset.events[ev][0] for ev in event_list}

        if intervals is None:
            used_intervals = {ev: dataset.events[ev][1] for ev in event_list}
        elif len(intervals) == 1:
            used_intervals = {ev: intervals[0] for ev in event_list}
        else:
            if len(event_list) != len(intervals):
                raise ValueError(
                    "Intervals should be the same number of events")
            used_intervals = {
                ev: intervals for ev, interval in zip(event_list, intervals)
            }

        # extract minor events, all the minor events should be pre-defined in the dataset
        minor_event_list = list(dataset.minor_events.keys())
        used_minor_events = {
            ev: dataset.minor_events[ev][0] for ev in minor_event_list}

        if minor_event_intervals is None:
            used_minor_intervals = {
                ev: dataset.minor_events[ev][1] for ev in minor_event_list}
        elif len(minor_event_intervals) == 1:
            used_minor_intervals = {ev: minor_event_intervals[0] for ev in minor_event_list}
        else:
            if len(event_list) != len(intervals):
                raise ValueError(
                    "Intervals should be the same number of events"
                )
            used_minor_intervals = {
                ev: intervals for ev, interval in zip(minor_event_list, minor_event_intervals)
            }

        encode_dict = dataset.encode
        encode_loop = dataset.encode_loop

        return used_events, used_intervals, used_minor_events, used_minor_intervals, encode_loop, encode_dict

    def register_trial_hook(self, hook):
        """Register trial hook before trial operation.

        Parameters
        __________
        hook : callable object to process Raw object before epoch operation.
            Different from the raw_hook, the trial hook allows you to do some specific operation
            BEFORE epoch operation (i.e. smallest encode unit) and AFTER raw continuous data operation

            Its signature should look like:

            hook(raw, caches) -> raw, caches

            where caches is a dict storing information, raw is MNE Raw instance

        Returns
        -------

        """
        self._trial_hook = hook

    def unregister_trial_hook(self):
        self._trial_hook = None

    @verbose
    def _get_single_subject_data(self, dataset, subject_id, verbose=False):
        """

        Parameters
        ----------
        dataset
        subject_id
        verbose

        Returns
        -------

        """

        used_events, used_intervals, used_minor_events, \
            used_minor_intervals, encode_loop, encode_dict = \
            self._map_events_intervals(dataset)

        # interval equally verification
        intervals = list(used_minor_intervals.values())
        if intervals.count(intervals[0]) == len(intervals):
            epoch_tmin = intervals[0][0]
            epoch_tmax = intervals[0][1]
        else:
            raise ValueError(
                'The defined intervals of minor event do not equal, please check')

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
                    elif hasattr(dataset, "raw_hook"):
                        raw, caches = dataset.raw_hook(raw, caches)

                    # pick selected channels by order
                    channels = (
                        dataset.channels
                        if self.select_channels is None
                        else self.select_channels
                    )
                    picks = pick_channels(raw.ch_names, channels, ordered=True)

                    stim_channels = mne.utils._get_stim_channel(
                        None, raw.info, raise_error=False
                    )

                    if len(stim_channels) > 0:
                        events = mne.find_events(
                            raw, shortest_event=0, initial_event=True
                        )
                    else:
                        events, _ = mne.events_from_annotations(
                            raw, event_id=(lambda x: int(x))
                        )

                    # extract main events
                    main_events = mne.pick_events(
                        events, include=list(used_events.values())
                    )

                    for event_name in used_events.keys():
                        # mne.pick_events returns any matching events in include
                        # only raise Runtime Error when nothing is found
                        # then we just skip this event
                        try:
                            selected_events = mne.pick_events(
                                events, include=used_events[event_name]
                            )
                        except RuntimeError:
                            continue

                        # Find trial_index in the original events series
                        trial_index = list(np.argwhere(
                            main_events[:, -1] == selected_events[0, 2]
                        ))
                        selected_annots = mne.annotations_from_events(
                            selected_events, sfreq=raw.info['sfreq'])
                        selected_annots.set_durations(
                            used_intervals[event_name][1] - used_intervals[event_name][0])

                        unit_raws = raw.copy().crop_by_annotations(annotations=selected_annots)

                        try:
                            unit_encode = encode_dict[event_name]
                        except Exception:
                            raise Exception(
                                "Dataset does not contain the encode key {:s}".format(
                                    event_name)
                            )

                        if isinstance(encode_loop, dict):
                            try:
                                encode_loop_size = encode_loop[event_name]
                            except Exception:
                                raise Exception(
                                    "Dataset does not contain the encode key {:s}".format(
                                        event_name)
                                )
                        elif isinstance(encode_loop, int):
                            encode_loop_size = encode_loop
                        else:
                            raise TypeError(
                                "Unknown encode_loop type"
                            )

                        for unit_raw in unit_raws:
                            # do trial hook
                            if self._trial_hook:
                                unit_raw, caches = self._trial_hook(
                                    unit_raw, caches)
                            elif hasattr(dataset, "epochs_hook"):
                                unit_raw, caches = dataset.trial_hook(
                                    unit_raw, caches)

                            # Try to extract minor events
                            minor_events = mne.find_events(
                                unit_raw, shortest_event=0, initial_event=True
                            )
                            minor_events = np.delete(minor_events, 0, axis=0)
                            selected_minor_events = mne.pick_events(minor_events,
                                                                    include=list(used_minor_events.values()))

                            # transform Raw to Epochs
                            epochs = mne.Epochs(
                                unit_raw,
                                selected_minor_events,
                                event_id=used_minor_events,
                                event_repeated="drop",
                                tmin=epoch_tmin,
                                tmax=epoch_tmax - 1.0 / unit_raw.info['sfreq'],
                                picks=picks,
                                proj=False,
                                baseline=None,
                                preload=True,
                                on_missing='ignore'
                            )

                            # skip invalid time intervals
                            if len(epochs) == 0:
                                continue

                            # check if the len of epochs matches with setting parameters
                            if epochs.__len__() != len(unit_encode) * encode_loop_size:
                                raise RuntimeError(
                                    "The setting parameters does not match the Epoch length"
                                )

                            # do epochs hook
                            if self._epochs_hook:
                                epochs, caches = self._epochs_hook(
                                    epochs, caches)
                            elif hasattr(dataset, "epochs_hook"):
                                epochs, caches = dataset.epochs_hook(
                                    epochs, caches)

                            # Get all epochs within a single 'character' event.
                            unit_X = epochs.get_data() * 1e6
                            unit_y = epochs.events[:, -1]
                            # trial_id is the index in the original event series of raw
                            # for the time encode paradigms, the trial_id indicate the index of main events
                            trial_id = trial_index[0]
                            trial_index.pop(0)
                            # Unlike the base paradigm class, we manually process a single trial
                            # So the meta only contains a single trial info
                            meta = pd.DataFrame(
                                {
                                    "subject": [subject],
                                    "session": [session],
                                    "run": [run],
                                    "event": [event_name],
                                    "trial_id": trial_id,
                                    "dataset": [dataset.dataset_code],
                                    "code": [unit_encode]
                                }
                            )

                            if self._data_hook:
                                unit_X, unit_y, meta, caches = self._data_hook(
                                    unit_X, unit_y, meta, caches)
                            elif hasattr(dataset, "data_hook"):
                                unit_X, unit_y, meta, caches = dataset.data_hook(
                                    unit_X, unit_y, meta, caches)

                            # collecting data
                            pre_X = Xs.get(event_name)
                            if pre_X is not None:
                                Xs[event_name].append(unit_X)
                            else:
                                Xs[event_name] = list()
                                Xs[event_name].append(unit_X)

                            pre_y = ys.get(event_name)
                            if pre_y is not None:
                                ys[event_name].append(unit_y)
                            else:
                                ys[event_name] = list()
                                ys[event_name].append(unit_y)

                            pre_meta = metas.get(event_name)
                            if pre_meta is not None:
                                metas[event_name] = pd.concat(
                                    (pre_meta, meta), axis=0, ignore_index=True
                                )
                            else:
                                metas[event_name] = meta
        return Xs, ys, metas

    @verbose
    def get_data(
            self,
            dataset: BaseTimeEncodingDataset,
            subjects: List[Union[int, str]] = [],
            return_concat: bool = False,
            n_jobs: int = -1,
            verbose: Optional[bool] = None,
    ):
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm. Check your events and channels settings".format(
                    dataset.dataset_code
                )
            )

        used_events, used_intervals, used_minor_events, \
            used_minor_intervals, encode_loop, encode_dict = \
            self._map_events_intervals(dataset)

        Xs = []
        ys = []
        metas = {}

        # Need to sort here
        # due to the subject data are storage in list in sequence
        subjects.sort()

        X, y, meta = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(self._get_single_subject_data)(
                    dataset, sub_id, verbose=verbose)
                for sub_id in subjects
            )
        )

        for event_name in used_events.keys():
            for i in range(len(subjects)):
                if event_name in X[i]:
                    for j in range(len(X[i][event_name])):
                        Xs.append(X[i][event_name][j])

            for i in range(len(subjects)):
                if event_name in y[i]:
                    for j in range(len(y[i][event_name])):
                        ys.append(y[i][event_name][j])

            if event_name in meta[i]:
                metas[event_name] = pd.concat(
                    [
                        meta[i][event_name]
                        for i in range(len(subjects))
                        if event_name in meta[i]
                    ],
                    axis=0,
                    ignore_index=True
                )

        metas = pd.concat(list(metas.values()), axis=0, ignore_index=True)

        return Xs, ys, metas
