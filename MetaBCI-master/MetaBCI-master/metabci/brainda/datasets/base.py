# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/6/01
# License: MIT License
"""
Basic elements to describe a BCI dataset.

Modified from https://github.com/NeuroTechX/moabb
"""
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

from mne.io import Raw
from mne.utils import verbose


class BaseDataset(metaclass=ABCMeta):
    """BaseDataset for all datasets."""

    def __init__(
            self,
            dataset_code: str,
            subjects: List[Union[int, str]],
            events: Dict[str, Tuple[Union[int, str], Tuple[float, float]]],
            channels: List[str],
            srate: Union[float, int],
            paradigm: str,
    ):
        """Parameters required for all datasets.

        Parameters
        ----------
        dataset_code : str
            unique identifier for dataset
        subjects : List[Union[int, str]]
            list of available subjects, could be int or str
        events : Dict[str, Tuple[Union[int, str], Tuple[float, float]]]
            describe events in the current dataset, the format is:

            {
                event_name: (event_id, (tmin, tmax))
            }
            event_name should be str, it could be anything, recommended names including:
            - left_hand
            - right_hand
            - hands
            - feet
            - rest
            - tongue
            event_id is the label in your experiments, should be str or int
            (tmin, tmax) is the task interval in senconds, tmin is the start time before event,
            tmax is the end time after event
        channels : List[str]
            available channels in the dataset, uppercase recommended
        srate : Union[float, int]
            sampling rate
        paradigm : str
            what kind of dataset this is, currently supported paradigms including:
            - p300
            - imagery
            - ssvep
            - ssavep
        """
        self.dataset_code = dataset_code
        self.subjects = subjects
        self.events = events
        self.channels = [ch.upper() for ch in channels]
        self.srate = srate
        self.paradigm = paradigm

    @abstractmethod
    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        """Get path to local copy of a subject data.

        Parameters
        ----------
        subject : Union[str, int]
            subject id
        path : Optional[Union[str, Path]], optional
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset_code)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset is not found under the given path,
            the data will be automatically downloaded to the specified folder,
            by default None
        force_update : bool, optional
            force update of the dataset even if a local copy exists,
            by default False
        update_path : Optional[bool], optional
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted,
            by default None
        proxies: Optional[Union[bool, str, int]], optional
            proxies if needed
        verbose : Optional[Union[bool, str, int]], optional
            [description], by default None

        Returns
        -------
        List[List[Union[str, Path]]]
            local path of a subject data, the first list is session and the second list is run
        """
        pass

    @abstractmethod
    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        """Get data of a subject.

        Parameters
        ----------
        subject : Union[str, int]
            subject id
        verbose : Optional[Union[bool, str, int]], optional
            [description], by default None

        Returns
        -------
        Dict[str, Dict[str, Raw]]
            {'sessio_id': {'run_id': Raw}}
        """
        pass

    @verbose
    def get_data(
            self,
            subjects: List[Union[int, str]],
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> Dict[Union[int, str], Dict[str, Dict[str, Raw]]]:
        """Get raw data.

        Parameters
        ----------
        subjects : List[Union[int, str]]
            subjects whose data should be returned

        Returns
        -------
        Dict[Union[int, str], Dict[str, Dict[str, Raw]]]
            returned raw ata, structured as
            {
                subject_id: {'sessio_id': {'run_id': Raw}}
            }

        Raises
        ------
        ValueError
            raise error if a subject is not valid
        """
        # use default subjects if not provided
        if subjects is None:
            subjects = self.subjects

        data = dict()
        for subject in subjects:
            if subject not in self.subjects:
                raise ValueError("Invalid subject {} given".format(subject))
            data[subject] = self._get_single_subject_data(subject)
        return data

    def __str__(self):
        event_info = "\n".join(
            [
                "    {}: {}".format(event_name, self.events[event_name])
                for event_name in self.events
            ]
        )
        desc = """Dataset {:s}:\n  Subjects  {:d}\n  Srate     {:.1f}\n  Events   \n{}\n  Channels  {:d}\n""".format(
            self.dataset_code,
            len(self.subjects),
            self.srate,
            event_info,
            len(self.channels),
        )
        return desc

    def __repr__(self):
        return self.__str__()

    def download_all(
            self,
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ):
        """Download all files.

        Parameters
        ----------
        path : Optional[Union[str, Path]], optional
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset_code)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset is not found under the given path,
            the data will be automatically downloaded to the specified folder, by default None
        force_update : bool, optional
            force update of the dataset even if a local copy exists, by default False
        proxies: Optional[Union[bool, str, int]], optional
            proxies if needed
        verbose : Optional[Union[bool, str, int]], optional
            [description], by default None
        """
        for subject in self.subjects:
            self.data_path(
                subject,
                path=path,
                proxies=proxies,
                force_update=force_update,
                update_path=True,
            )


class BaseTimeEncodingDataset(BaseDataset):
    def __init__(self,
                 dataset_code: str,
                 subjects: List[Union[int, str]],
                 events: Dict[str, Tuple[Union[int, str], Tuple[float, float]]],
                 channels: List[str],
                 srate: Union[float, int],
                 paradigm: str,
                 minor_events: Dict[str, Tuple[Union[int, str], Tuple[float, float]]],
                 encode: Dict[str, List[Union[int, str]]],
                 encode_loop: int):
        super(BaseTimeEncodingDataset, self).__init__(
            dataset_code=dataset_code,
            subjects=subjects,
            events=events,
            channels=channels,
            srate=srate,
            paradigm=paradigm
        )
        self.minor_events = minor_events
        self.encode = encode
        self.encode_loop = encode_loop

    @abstractmethod
    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        pass

    @abstractmethod
    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        pass
