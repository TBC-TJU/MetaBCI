# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/30
# License: MIT License
"""
Brain/Neuro Computer Interface (BNCI) datasets.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

import numpy as np
import mne
from mne.io import Raw, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat

BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


class BNCI2014001(BaseDataset):
    """BNCI 2014-001 Motor Imagery dataset.

    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**

    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left.  All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------

    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.
    """

    _EVENTS = {
        "left_hand": (1, (2, 6)),
        "right_hand": (2, (2, 6)),
        "feet": (3, (2, 6)),
        "tongue": (4, (2, 6)),
    }

    _CHANNELS = [
        "FZ",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "P1",
        "PZ",
        "P2",
        "POZ",
    ]

    def __init__(self):
        super().__init__(
            dataset_code="bnci2014001",
            subjects=list(range(1, 10)),
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
        base_url = "{:s}001-2014/A{:02d}".format(BNCI_URL, subject)

        dests = [
            [
                mne_data_path(
                    "{:s}{:s}.mat".format(base_url, "E"),
                    "bnci",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ],
            [
                mne_data_path(
                    "{:s}{:s}.mat".format(base_url, "T"),
                    "bnci",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ],
        ]
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
            run_arrays = loadmat(run_dests[0])["data"]
            runs = dict()
            for irun, run_array in enumerate(run_arrays):
                X = run_array.X.T * 1e-6  # volt
                trial = run_array.trial
                y = run_array.y
                stim = np.zeros((1, X.shape[-1]))

                if y.size > 0:
                    stim[0, trial - 1] = y

                data = np.concatenate((X, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + [
                    "EOG1",
                    "EOG2",
                    "EOG3",
                ]
                ch_types = ["eeg"] * len(self._CHANNELS) + ["eog"] * 3
                ch_names = ch_names + ["STI 014"]
                ch_types = ch_types + ["stim"]

                info = mne.create_info(ch_names, self.srate, ch_types=ch_types)
                raw = RawArray(data, info)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess


class BNCI2014004(BaseDataset):
    """BNCI 2014-004 Motor Imagery dataset.

    Dataset B from BCI Competition 2008.

    **Dataset description**

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------

    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
           G. Pfurtscheller. Brain-computer communication: motivation, aim,
           and impact of exploring a virtual apartment. IEEE Transactions on
           Neural Systems and Rehabilitation Engineering 15, 473–482, 2007

    """

    _EVENTS = {
        "left_hand": (1, (3, 7.5)),
        "right_hand": (2, (3, 7.5)),
    }

    _CHANNELS = ["C3", "CZ", "C4"]

    def __init__(self):
        super().__init__(
            dataset_code="bnci2014004",
            subjects=list(range(1, 10)),
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
        base_url = "{:s}004-2014/B{:02d}".format(BNCI_URL, subject)

        # actually 5 sessions, be careful
        dests = [
            [
                mne_data_path(
                    "{:s}{:s}.mat".format(base_url, "E"),
                    "bnci",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ],
            [
                mne_data_path(
                    "{:s}{:s}.mat".format(base_url, "T"),
                    "bnci",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ],
        ]
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

        sess_arrays = np.append(loadmat(dests[0][0])["data"], loadmat(dests[1][0])["data"])

        sess = dict()
        for isess, sess_array in enumerate(sess_arrays):
            runs = dict()
            X = (sess_array.X).T * 1e-6  # volt
            trial = sess_array.trial
            y = sess_array.y
            stim = np.zeros((1, X.shape[-1]))

            if y.size > 0:
                stim[0, trial - 1] = y

            data = np.concatenate((X, stim), axis=0)

            ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + [
                "EOG1",
                "EOG2",
                "EOG3",
            ]
            ch_types = ["eeg"] * len(self._CHANNELS) + ["eog"] * 3
            ch_names = ch_names + ["STI 014"]
            ch_types = ch_types + ["stim"]

            info = mne.create_info(ch_names, self.srate, ch_types=ch_types)
            raw = RawArray(data, info)
            raw = upper_ch_names(raw)
            raw.set_montage(montage)
            runs["run_0"] = raw
            sess["session_{:d}".format(isess)] = runs
        return sess
