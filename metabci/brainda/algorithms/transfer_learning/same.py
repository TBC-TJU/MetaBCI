# -*- coding: utf-8 -*-

"""source aliasing matrix estimation (SAME) and its multi-stimulus version (msSAME).

A data augmentation method named Source Aliasing Matrix Estimation
(SAME) [1] to enhance the performance of state-of-the-art spatial filtering methods (i.e., eTRCA, TDCA)
for SSVEP-BCIs. Based on the superposition model of SSVEPs, the task-related components are reconstructed
by estimating the source aliasing matrixes. After adding noise, multiple artificial signals are generated
and then added to calibrated data in an appropriate proportion.

In 2023, paper [2] proposes an extended version of SAME, called multi-stimulus SAME (msSAME), which exploits
the similarity of the aliasing matrix across frequencies to enhance the performance of SSVEP-BCI with
insufficient calibration trials.

souce code of SAME: https://github.com/RuixinLuo/Source-Aliasing-Matrix-Estimation-DataAugmentation-SAME-SSVEP

.. [1] Luo R., et al. Data augmentation of SSVEPs using source aliasing matrix estimation for
       brain-computer interfaces. IEEE Trans. Biomed. Eng., 2022. DOI: 10.1109/TBME.2022.3227036
.. [2] Luo R., et al. Almost free of calibration for SSVEP-based brain-computer interfaces.
       Journal of Neural Engineering, 2023. DOI: 10.1088/1741-2552/ad0b8f
"""

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from .lst import lst_kernel


def TRCs_estimation(data, mean_target):
    """source signal estimation using LST [1]_

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-01-09

    update log:
        2023-09-06 by Ruixin Luo <ruixin_luo@tju.edu.cn>

        2023-12-09 by heoohuan <heoohuan@163.com>（Modify code annotation）

    Parameters
    ----------
    data:ndarray
        Reference signal, shape(n_channel_1, n_times).
    mean_target:ndarray
        Average template, shape(n_channel_2, n_times).

    Returns
    -------
    data_after:ndarray
        Source signal, shape(n_channel_2, n_times).

    References
    ----------
    .. [1] Chiang, K. J., Wei, C. S., Nakanishi, M., & Jung, T. P. (2021, Feb 11) .
       Boosting  template-based ssvep decoding by cross-domain transfer learning.
       J Neural Eng, 18(1), 016002.

    """

    X_a = data
    X = mean_target

    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X_a, T=X)
    # source signal estimation
    data_after = PT @ X_a

    return data_after


def get_augment_noiseAfter(fs, f, Nh, n_Aug, mean_temp, alpha=0.05):
    """Artificially generated signals by SAME

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-01-09

    update log:
        2023-09-06 by Ruixin Luo <ruixin_luo@tju.edu.cn>

    Parameters
    ----------
    fs : int
        Sampling rate.
    f : float
        Frequency of signal.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals.
    mean_temp: ndarray
        Average template, shape(n_channels, n_times).
    alpha: float
        Intensity of noise, default 0.05.

    Returns
    -------
    data_aug : ndarray
        Artificially generated signals, shape(n_channel, n_times, n_Aug).

    Note
    ----
    Please note that we apply SAME before filter bank analysis in the MetaBCI version.
    This is convenient for compatibility with MetaBCI and saves computational effort.
    After testing, it still has a similar improvement effect.

    """

    nChannel, nTime = mean_temp.shape
    #  Generate reference signal Yf
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((nTime, 2 * Nh))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2 + 1] = y_cos

    Z = TRCs_estimation(Yf.T, mean_temp)
    # get vars of Z
    vars_z = np.diag(np.var(Z, -1))

    # add noise
    data_aug = np.zeros((nChannel, nTime, n_Aug))
    for i_aug in range(n_Aug):
        # Randomly generated noise
        Datanoise = np.random.multivariate_normal(
            mean=np.zeros(nChannel), cov=vars_z, size=nTime)
        data_aug[:, :, i_aug] = Z + alpha * Datanoise.T

    return data_aug


class SAME(BaseEstimator, TransformerMixin):
    """
    source aliasing matrix estimation (SAME) [1]_.

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-01-09

    update log:
        2023-09-06 by Ruixin Luo <ruixin_luo@tju.edu.cn>

        2023-10-03 by Jie Mei <chmeijie@tju.edu.cn>

    Parameters
    ----------
    fs : int
        Sampling rate.
    flist : list
        Frequency of all class.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals.
    alpha: float
        Intensity of noise, default 0.05.

    Attributes
    ----------
    T_ : list
        Average template for different classes of data.
    classes_ : ndarray
        number of classes.

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Luo R., et al. Data augmentation of SSVEPs using source aliasing matrix
       estimation for brain-computer interfaces. IEEE Trans. Biomed. Eng.,
       2022. DOI: 10.1109/TBME.2022.3227036

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using SAME

        from metabci.brainda.algorithms.transfer_learning import SAME
        same = SAME(fs = 250, Nh = 5, flist = freq_list, n_Aug = 4)
        same.fit(X_train , y_train)
        X_aug, y_aug = same.augment()
        X_train_new = np.concatenate((X_train, X_aug), axis=0)
        y_train_new = np.concatenate((y_train, y_aug), axis=0)

    """

    def __init__(self,
                 n_jobs=None,
                 fs=250,
                 flist=None,
                 Nh=5,
                 n_Aug=5,
                 alpha=0.05):
        self.n_jobs = n_jobs
        self.fs = fs
        self.Nh = Nh
        self.n_Aug = n_Aug
        self.flist = flist
        self.alpha = alpha

    def fit(self, X: ndarray, y: ndarray):
        """ Model training.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,)

        """
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y == label], axis=0) for label in self.classes_]
        return self

    def augment(self):
        """ Calculating augmentation signals.

        Returns
        -------
        X_aug: ndarray
            augmentation data, shape(n_events*n_aug, n_channels, n_samples).
        y_aug: ndarray
            Label of augmentation data, shape(n_events*n_aug,).
        """

        X_aug = []
        y_aug = []
        for i, label in enumerate(self.classes_):
            temp = self.T_[i]
            f = self.flist[i]
            data_aug = get_augment_noiseAfter(
                fs=self.fs,
                f=f,
                Nh=self.Nh,
                n_Aug=self.n_Aug,
                mean_temp=temp,
                alpha=self.alpha)
            # n_aug, n_channel, n_times
            data_aug = np.transpose(data_aug, [2, 0, 1])
            X_aug.append(data_aug)
            y_aug.append(np.ones(self.n_Aug, dtype=np.int32) * label)

        X_aug = np.concatenate(X_aug, axis=0)
        y_aug = np.concatenate(y_aug, axis=0)
        return X_aug, y_aug


def get_augment_noiseAfter_ms(fs, f_list, phi_list, Nh, n_Aug, mean_temp_all, iEvent, n_Templates, alpha=0.05):
    """Artificially generated signals by msSAME.

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-11-09

    update log:
        2023-11-09 by Ruixin Luo <ruixin_luo@tju.edu.cn>

        2023-12-09 by heoohuan <heoohuan@163.com>（Modify code annotation）


    Parameters
    ----------
    fs : int
        Sampling rate.
    f_list : list
        The all frequency of reference signal.
    phi_list: list
        The all phase of reference signal.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals
    mean_temp_all: ndarray-like (n_channel, n_times, n_events)
        Average template of all events.
    iEvent: int
        the i-th event for the selection of neighboring frequencies.
    n_Templates: int
        The number of neighboring frequencies.
    alpha: float
        Intensity of noise, default 0.05.

    Returns
    -------
    data_aug : ndarray-like (n_channel, n_times, n_Aug)
        Artificially generated signals.

    Note
    ----
    Please note that we apply msSAME before filter bank analysis in the MetaBCI version.
    This is convenient for compatibility with MetaBCI and saves computational effort.
    After testing, it still has a similar improvement effect.

    """
    Nh = Nh
    mean_temp_all = np.transpose(mean_temp_all, [1, 0, 2])

    [nTimes, nChannels, nEvents] = mean_temp_all.shape

    if n_Templates == 0 or n_Templates == 1:  # Original SAME
        template_st = iEvent
        template_ed = iEvent + 1
        n_Templates = 1
    else:  # Multi-stimulus SAME
        d0 = int(n_Templates / 2)
        d1 = nEvents

        n = iEvent + 1
        if n <= d0:
            template_st = 1
            template_ed = n_Templates
        elif (n > d0) & (n < (d1 - d0 + 1)):
            template_st = n - d0
            template_ed = n + (n_Templates - d0 - 1)
        else:
            template_st = (d1 - n_Templates + 1)
            template_ed = d1
        template_st = int(template_st - 1)
        template_ed = int(template_ed)

    #  Concatenation of the templates (or sine-cosine references)
    ms_ref = np.zeros((n_Templates * nTimes, 2 * Nh))
    ms_template = np.zeros((n_Templates * nTimes, nChannels))

    index = 0
    for j in range(template_st, template_ed, 1):
        # sine-cosine references
        f = f_list[j]
        phi = phi_list[j]
        Ts = 1 / fs
        n = np.arange(nTimes) * Ts
        Yf = np.zeros((nTimes, Nh * 2))
        for iNh in range(Nh):
            y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
            Yf[:, iNh * 2] = y_sin
            y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
            Yf[:, iNh * 2 + 1] = y_cos
        ms_ref[index * nTimes: (index + 1) * nTimes, :] = Yf
        # templates
        ss = mean_temp_all[:, :, j]
        # ss = ss - np.tile(np.mean(ss, 0), (ss.shape[0], 1))
        ms_template[index * nTimes:(index + 1) * nTimes, :] = ss
        index = index + 1

    PT = lst_kernel(S=ms_ref.T, T=ms_template.T)

    # chount Z
    # sine-cosine references at target frequency
    f = f_list[iEvent]
    phi = phi_list[iEvent]
    Ts = 1 / fs
    n = np.arange(nTimes) * Ts
    Yf = np.zeros((nTimes, Nh * 2))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
        Yf[:, iNh * 2 + 1] = y_cos

    Z = PT @ Yf.T

    # get vars of Z
    vars_z = np.diag(np.var(Z, -1))

    # add noise
    data_aug = np.zeros((nChannels, nTimes, n_Aug))
    for i_aug in range(n_Aug):
        # Randomly generated noise
        Datanoise = np.random.multivariate_normal(mean=np.zeros(nChannels), cov=vars_z, size=nTimes)
        data_aug[:, :, i_aug] = Z + alpha * Datanoise.T

    return data_aug


class MSSAME(BaseEstimator, TransformerMixin):
    """
    multi-stimulus source aliasing matrix estimation (msSAME) [1]_.

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-11-13

    update log:
        2023-11-13 by Ruixin Luo <ruixin_luo@tju.edu.cn>

        2023-12-09 by heoohuan <heoohuan@163.com>（Modify code annotation）


    Parameters
    ----------
    fs : int
        Sampling rate.
    flist : list
        Frequency of all class.
    plist : list
        Phase of all class.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals
    n_Neig: int
        The number of neighborhood frequency
    alpha: float
        Intensity of noise, default 0.05.

    Attributes
    ----------
    T_ : list
        Average template for different classes of data.
    classes_ : ndarray
        number of classes.

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Luo R., et al. Almost free of calibration for SSVEP-based brain-computer interfaces.
       Journal of Neural Engineering, 2023. DOI: 10.1088/1741-2552/ad0b8f

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using MSSAME

        from metabci.brainda.algorithms.transfer_learning import MSSAME
        mssame = MSSAME(fs = 250, Nh = 5, flist = freq_list, plist=phase_list, n_Aug=4, n_Neig=14)
        mssame.fit(X_train , y_train)
        X_aug, y_aug = mssame.augment()
        X_train_new = np.concatenate((X_train, X_aug), axis=0)
        y_train_new = np.concatenate((y_train, y_aug), axis=0)

    """

    def __init__(self, n_jobs=None, fs=250, flist=None, plist=None, Nh=5, n_Aug=5, n_Neig=12, alpha=0.05):
        self.n_jobs = n_jobs
        self.fs = fs
        self.Nh = Nh
        self.n_Aug = n_Aug
        self.flist = flist
        self.plist = plist
        self.n_Neig = n_Neig
        self.alpha = alpha

    def fit(self, X: ndarray, y: ndarray):
        """ model training

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,)

        """
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y == label], axis=0) for label in self.classes_]
        return self

    def augment(self):
        """ Calculating augmentation signals.

        Returns
        -------
        X_aug: ndarray
            augmentation data, shape(n_events*n_aug, n_channels, n_samples).
        y_aug: ndarray
            Label of augmentation data, shape(n_events*n_aug,).
        """

        X_aug = []
        y_aug = []
        n_channel, n_times = np.shape(self.T_[0])
        n_event = len(self.classes_)
        temp = np.zeros((n_channel, n_times, n_event))  # n_channel * n_times * n_events
        for n in range(n_event):
            temp[:, :, n] = self.T_[n]
        # generated signals
        for i, label in enumerate(self.classes_):
            data_aug = get_augment_noiseAfter_ms(fs=self.fs, f_list=self.flist, phi_list=self.plist,
                                                 Nh=self.Nh, n_Aug=self.n_Aug, mean_temp_all=temp, iEvent=i,
                                                 n_Templates=self.n_Neig, alpha=self.alpha)
            data_aug = np.transpose(data_aug, [2, 0, 1])  # n_aug, n_channel, n_times
            X_aug.append(data_aug)
            y_aug.append(np.ones(self.n_Aug, dtype=np.int32) * label)

        X_aug = np.concatenate(X_aug, axis=0)
        y_aug = np.concatenate(y_aug, axis=0)
        return X_aug, y_aug
