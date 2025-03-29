# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/29
# License: MIT License

from typing import Optional, List, cast
from functools import partial

import numpy as np
from scipy.linalg import eigh, pinv, qr
from scipy.stats import pearsonr
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
import math
import time

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC
from joblib import Parallel, delayed

from .base import FilterBankSSVEP


def _ged_wong(
    Z: ndarray,
    D: Optional[ndarray] = None,
    P: Optional[ndarray] = None,
    n_components=1,
    method="type1",
):
    if method != "type1" and method != "type2":
        raise ValueError("not supported method type")

    A = Z
    if D is not None:
        A = D.T @ A
    if P is not None:
        A = P.T @ A
    A = A.T @ A
    if method == "type1":
        B = Z
        if D is not None:
            B = D.T @ Z
        B = B.T @ B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == "type2":
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)

    D_exist = cast(ndarray, D)
    ind = np.argsort(D_exist)[::-1]
    D_exist, W = D_exist[ind], W[:, ind]
    return D_exist[:n_components], W[:, :n_components]


def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    Z = X.T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    V = pinv(R) @ Q.T @ X.T @ U  # V for Yf
    return U, V


def _scca_feature(X: ndarray, Yf: ndarray, n_components: int = 1):
    rhos = []
    for Y in Yf:
        U, V = _scca_kernel(X, Y)
        a = U[:, :n_components].T @ X
        b = V[:, :n_components].T @ Y
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Standard CCA (sCCA).The Canonical Correlation Analysis (CCA) method finds the coefficients of the linear combination
    between the test signal and the Fourier series reference signal for a given frequency-periodic signal to find the
    maximum correlation between the two sets of signals. To identify the frequency of the SSVEP, CCA calculates the
    typical correlation between the multichannel SSVEP and the reference signal corresponding to each stimulus frequency,
    and the frequency of the reference signal with the largest correlation is regarded as the frequency of the
    SSVEP[1]_[2]_.SCCA is the standard CCA method.

    Parameters
    ----------
    n_components : int
        The number of feature dimensions after dimensionality reduction,
        the dimension of the spatial filter, defaults to 1.
    n_jobs : int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    Yf_ : ndarray
        The reference signal provided, defaults to None.

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Lin Z, Zhang C, Wu W, et al. Frequency recognition based on canonical correlation analysis for
        SSVEP-based BCIs[J].IEEE transactions on biomedical engineering, 2006, 53(12): 2610-2614.

    .. [2] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer
        interface[J].Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.

    Tip
    ----
    .. code-block:: python
       :caption: A example using SCCA

        from metabci.brainda.algorithms.decomposition.cca import SCCA
        estimator = SCCA()
        p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])

    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        Yf: Optional[ndarray] = None,
    ):
        """model training

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,)
        Yf: ndarray
            Sine and cosine reference signal, shape(n_classes, 2*n_harmonics, n_samples).
        """
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        return self

    def transform(self, X: ndarray):
        """The correlation coefficients of the signals from different trials were obtained by converting X
        into features.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            he correlation coefficients, shape(n_trials, n_fre)

        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Yf = self.Yf_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Yf) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X : ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels : ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)
        return labels


class FBSCCA(FilterBankSSVEP, ClassifierMixin):
    """
    Filter bank SCCA methods, i.e., SCCA methods that combine the application of multiple filters in order to decompose
    the SSVEP signal into specific subbands[1]_ .This class is a FBSCCA classifier.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    filterweights: ndarray
        Filter weights, defaults to None.
    n_jobs: int
        The number of CPU working cores, default is None.

    References
    ----------
    .. [1] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed
        SSVEP-based brain–computer interface[J]. Journal of neural engineering, 2015, 12(4): 046008.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using FBSCCA

       import sys
       import numpy as np
       from brainda.algorithms.decomposition import FBSCCA
       from brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
       wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
       ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
       filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)
       filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
       estimator = FBSCCA(filterbank=filterbank,n_components=1,filterweights=np.array(filterweights),n_jobs=-1)
       accs = []
       for k in range(kfold):
           train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
           # merge train and validate set
           train_ind = np.concatenate((train_ind, validate_ind))
           p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])
           accs.append(np.mean(p_labels==y[test_ind]))
           print(np.mean(accs))
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            SCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = np.argmax(features, axis=-1)
        return labels


def _itcca_feature(
    X: ndarray,
    templates: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
    method: str = "itcca1",
):
    """
    ItCCA feature extraction
    """
    rhos = []
    if method == "itcca1":
        for Xk in templates:
            U, V = _scca_kernel(X, Xk)
            a = U[:, :n_components].T @ X
            b = V[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    elif method == "itcca2":
        Us = cast(ndarray, Us)
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class ItCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The Individual Template-based Canonical Correlation Analysis (It-CCA) method is an extension of the CCA method in
    which the reference signal is a VEP template obtained by averaging multiple EEG trials from each individual's
    calibration data, and the individual SSVEP training data is used in the CCA method to improve the frequency detection
    of SSVEP [1]_.This class is a itCCA classifier

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    Yf_: ndarray
        Reference signal.
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Test data after spatial filtering
    Us_: ndarray
        Spatial filter
    Vs_: ndarray
        Spatial filter

    References
    ----------
    .. [1] Brogin J A F, Faber J, Bueno D D. Enhanced use practices in SSVEP-based BCIs using an analytical approach of
        canonical correlation analysis[J]. Biomedical Signal Processing and Control, 2020, 55: 101644.
    """

    def __init__(
        self,
        n_components: int = 1,
        method: str = "itcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        if self.method == "itcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )
        if self.method == "itcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = cast(ndarray, Yf)
            self.Us_, self.Vs_ = zip(
                *[
                    _scca_kernel(self.templates_[i], self.Yf_[i])
                    for i in range(len(self.classes_))
                ]
            )
            self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        """Transform the X into features and calculate the correlation coefficients of different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            Correlation coefficients, shape(n_trials, n_fre).

        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        method = self.method
        Us = None
        if method == "itcca2":
            Us = self.Us_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(_itcca_feature, Us=Us, n_components=n_components, method=method)
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBItCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank ItCCA method, i.e., the ItCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    filterweights: ndarray
        Filter weights, defaults to None.
    n_jobs: int
        The number of CPU working cores, default is None.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Bolaños M C, Ballestero S B, Puthusserypady S. Filter bank approach for enhancement of supervised Canonical
        Correlation Analysis methods for SSVEP-based BCI spellers[C]//2021 43rd Annual International Conference of
        the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021: 337-340.

    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "itcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ItCCA(n_components=n_components, method=method, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _mscca_feature(X: ndarray, templates: ndarray, U: ndarray, n_components: int = 1):
    rhos = []
    for Xk in zip(templates):
        a = U[:, :n_components].T @ X
        b = U[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class MsCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Since the sine-cosine signal may not be the ideal reference signal, the Multiset Canonical Correlation Analysis
    (MsetCCA) method uses joint spatial filtering of multiple sets of data to create an optimized reference signal that
    extracts common SSVEP features from multiple sets of EEG data recorded at the same stimulus frequency[1]_.
    Note: MsCCA heavily depends on Yf, thus the phase information should be included when designs Yf.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    Yf_: ndarray
        Reference signals
    Us_: ndarray
        Spatial filter
    Ts_: ndarray
        Spatial filter

    References
    ----------
    .. [1] Zhang YU, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical correlation
        analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.
    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.U_, self.V_ = _scca_kernel(
            np.concatenate(self.templates_, axis=-1), np.concatenate(self.Yf_, axis=-1)
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre).
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        U = self.U_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_mscca_feature, U=U, n_components=n_components))(
                a, templates
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBMsCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank MsetCCA method, i.e., the MsetCCA method that combines the application of multiple filters
    in order to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list.
    filterweights: ndarray
        Weights of filter banks
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Zhang Y U, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical correlation
        analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ecca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U3 in zip(templates, Yf, Us):
        rho_list = []
        # 14a, 14d
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14b
        U2, _ = _scca_kernel(X, Xk)
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14c
        a = U3[:, :n_components].T @ X
        b = U3[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """The Extended Canonical Correlation Analysis (eCCA) method combines the advantages of sCCA and itCCA while
    applying the individual averaging templates and the positive cosine reference signal correlation information,
    thus obtaining better recognition performance[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    Yf_: ndarray
        Reference signals.
    Us_: ndarray
        Spatial filter.
    Vs_: ndarray
        Spatial filter.

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer interface[J].
        Proceedings of the national academy of sciences. 2015. 112(44): E6058-E6067.
    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        """

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ecca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBECCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank eCCA method, i.e., an eCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands [1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Weights of filter bank
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Tong C, Wang H. A Novel Low Training Cost SSVEP Detector Design[C]//2021 14th International Symposium on
        Computational Intelligence and Design (ISCID). IEEE, 2021: 130-133.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using FBECCA

       import sys
       import numpy as np
       from brainda.algorithms.decomposition import FBECCA
       from brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
       wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
       ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
       filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)
       filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
       estimator = FBECCA(filterbank=filterbank,n_components=1,filterweights=np.array(filterweights),n_jobs=-1)
       accs = []
       for k in range(kfold):
            train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
            train_ind = np.concatenate((train_ind, validate_ind))
            p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])
            accs.append(np.mean(p_labels==y[test_ind]))
       print(np.mean(accs))
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ECCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ttcca_template(X: ndarray, y: ndarray, y_sub: Optional[ndarray] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    labels = np.unique(y)
    if y_sub is None:
        templates = np.stack([np.mean(X[y == label], axis=0) for label in labels])
    else:
        subjects = np.unique(y_sub)
        templates = 0
        for sub_id in subjects:
            templates += np.stack(
                [
                    np.mean(X[(y == label) & (y_sub == sub_id)], axis=0)
                    for label in labels
                ]
            )
        templates /= len(subjects)
    return templates


def _ttcca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U2 in zip(templates, Yf, Us):
        rho_list = []
        # rho1
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho3
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho2
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class TtCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The Transfer Template-based Canonical Correlation Analysis (tt-CCA) method migrates SSVEP templates from existing
    subjects to new subjects to enhance SSVEP detection. EEG templates were generated for the new subjects using the
    existing source subject dataset, i.e., migrating EEG templates to capture the frequency and phase information of
    SSVEP[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Individual average template signals.
    Yf_: ndarray
        Reference signals.
    Us_: ndarray
        Spatial filter.
    Vs_: ndarray
        Spatial filter.

    References
    ----------
    .. [1] Yuan P, Chen X, Wang Y, et al. Enhancing performances of SSVEP-based brain–computer interfaces via exploiting
        inter-subject information[J]. Journal of neural engineering, 2015, 12(4): 046006.

    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray, y_sub: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        y_sub: ndarray
            Existing source subject data
        """

        self.classes_ = np.unique(y)
        self.templates_ = _ttcca_template(X, y, y_sub=y_sub)

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ttcca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBTtCCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank TtCCA method, i.e., a TtCCA method that combines the application of multiple filters in order to
    decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Weights of filter banks
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.


    References
    ----------
    .. [1] Yuan P, Chen X, Wang Y, et al. Enhancing performances of SSVEP-based brain–computer interfaces via
        exploiting inter-subject information[J]. Journal of neural engineering, 2015, 12(4): 046006.


    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TtCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X: ndarray,  # type: ignore[override]
        y: ndarray,
        Yf: Optional[ndarray] = None,
        y_sub: Optional[ndarray] = None,
    ):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        y_sub: ndarray
            Existing source subject data.
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf, y_sub=y_sub)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _msetcca_kernel1(X: ndarray):
    """Multi-set CCA1 (MsetCCA1).

    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


def _msetcca_kernel2(X: ndarray, Xk: ndarray, Yf: ndarray):
    C, N = X.shape
    M = 3
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = block_diag([X, Yf, Xk]).T
    D, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U[:C, :], D


def _msetcca_feature2(
    X: ndarray, templates: ndarray, Yf: ndarray, n_components: int = 1
):
    feat = []
    for Xk, Y in zip(templates, Yf):
        U, D = _msetcca_kernel2(X, Xk, Y)
        A = U[:, :n_components].T @ X
        B = U[:, :n_components].T @ Xk
        rho = np.array([pearsonr(a, b)[0] for a, b in zip(A, B)])
        rho = D[0] * np.sign(rho) * (rho**2)
        feat.append(rho)
    feat = np.concatenate(feat, axis=-1)
    return feat


class MsetCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Since the sine-cosine signal may not be the ideal reference signal, the Multiset Canonical Correlation Analysis
    (MsetCCA) method uses joint spatial filtering of multiple sets of data to create an optimized reference signal that
    extracts common SSVEP features from multiple sets of EEG data recorded at the same stimulus frequency[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    methods: str
        Two Pattern Feature Extraction and Fitting Classifier Model Methods Judgment, defaulting to 'msetcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Template signals
    Yf_: ndarray
        Reference signals
    Us_: ndarray
        Spatial filter
    Ts_: ndarray
        Spatial filter

    References
    ----------
    .. [1]  Zhang YU, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical
        correlation analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.


    """

    def __init__(
        self,
        n_components: int = 1,
        method: str = "msetcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        if self.method == "msetcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        if self.method == "msetcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = Yf
            feat = self.transform(X)
            self.clf_ = SVC()
            self.clf_.fit(feat, y)
        elif self.method == "msetcca1":
            self.Us_, self.Ts_ = zip(
                *[_msetcca_kernel1(X[y == label]) for label in self.classes_]
            )
            self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        if self.method == "msetcca1":
            Ts = self.Ts_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_scca_feature, n_components=n_components))(a, Ts)
                for a in X
            )
            rhos = np.stack(rhos)
        elif self.method == "msetcca2":
            templates = self.templates_
            Yf = self.Yf_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_msetcca_feature2, n_components=n_components))(
                    a, templates, Yf
                )
                for a in X
            )
            rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        if self.method == "msetcca1":
            labels = self.classes_[np.argmax(feat, axis=-1)]
        elif self.method == "msetcca2":
            labels = self.clf_.predict(feat)
        return labels


class FBMsetCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank MsetCCA method, i.e., the MsetCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list.
    filterweights: ndarray
        Weights of filter banks.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    methods: str
        Two Pattern Feature Extraction and Fitting Classifier Model Methods Judgment, defaulting to 'msetcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Zhang Y U, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical
        correlation analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.

    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "msetcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCA(n_components=n_components, method=method),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _msetccar_kernel(X: ndarray, Yf: ndarray):
    """Multi-set CCA1 with reference signals (MsetCCA-R).
    Parameters
    ----------
    X: ndarray
        EEG data, shape(n_trials, n_channels, n_samples).
    Yf: ndarray
        Reference signal, shape(n_classes, 2*n_harmonics, n_samples)


    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = min(C, Yf.shape[0])
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


class MsetCCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = 1):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_, self.Ts_ = zip(
            *[
                _msetccar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components

        Ts = self.Ts_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Ts) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBMsetCCAR(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCAR(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _trca_kernel(X: ndarray):
    """TRCA spatial filter calculate.

    Parameters
    ----------
    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


def _trca_feature(
    X: ndarray,
    templates: ndarray,
    Us: ndarray,
    n_components: int = 1,
    ensemble: bool = True,
):
    rhos = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        for Xk in templates:
            a = U.T @ X
            b = U.T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return rhos


class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """The core idea of Task-Related Component Analysis (TRCA) algorithm is to extract task-related components by
    improving the repeatability between trials, specifically, the algorithm is based on inter-trial covariance matrix
    maximization to achieve the extraction of task-related components, which belongs to the supervised learning method[1]_.


    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.

    References
    ----------
    .. [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller using
        task-related component analysis. IEEE Transactions on Biomedical Engineering, 2018, 104-112.

    """

    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        self.Us_ = np.stack([_trca_kernel(X[y == label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank TRCA (filter bank Task-Related Component Analysis, fbTRCA) adds the filter bank analysis method
    to TRCA by combining the fundamental and harmonic components of the signal. The EEG signal is first filtered using
    multiple subband filters with different cutoff frequencies to obtain the subband filtered signal. Subsequently,
    the correlation coefficients of the subband signals are summed according to a weighting function, and finally this
    weighted correlation coefficient sum is used as the feature discriminant [1]_.


    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Filter weights, defaults to None.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.


    References
    ----------
    .. [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller
        using task-related component analysis.IEEE Transactsions on Biomedical Engineering, 2018, 104-112.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using FBTRCA

        import numpy as np
        from brainda.algorithms.decomposition import FBTRCA
        X = np.zeros((4,22,22))
        for i in range(4):
            X[i,...] = np.identity(22)*0.5 + np.random.normal(-1,3,(22,22))*2
        y = np.array([1,1,2,2])
        filterbank = [np.ones((3,6))]
        filterweights = np.array([[0.3, -0.1], [0.5, -0.1]])
        estimator = FBTRCA(filterbank=filterbank,n_components=1, ensemble=True,filterweights=np.array(filterweights),n_jobs=-1)
        p_labels = estimator.fit(X, y)
        print(estimator.predict(np.identity(22)))
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCA(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal, shape(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _trcar_kernel(X: ndarray, Yf: ndarray):
    """TRCA-R.
    X: (n_trials, n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


class TRCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The task-related component analysis algorithm with sine-cosine reference signal (TRCA with sine-cosine reference
    signal, TRCA-R) is based on the TRCA algorithm, and the main improvement point is to add the step of orthogonal
    projection of the signal to the subspace of sine-cosine template during the training process, which further
    extracts the components of the EEG signal that are more correlated with the sine-cosine fluctuations of SSVEP[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template.
    Yf_: ndarray
        Sine-Cosine reference signal.
    Us_: ndarray
        Spatial filters obtained for each class of training signals.


    References
    ----------
    .. [1] Wong C, Wang B, Wang Z, et al. Spatial Filtering in SSVEP-Based BCIs: Unified Framework and New
        Improvements. IEEE Transactions on Biomedical Engineering 2020, 3057-3072.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using TRCAR

        import numpy as np
        from brainda.algorithms.decomposition import TRCAR
        X = np.array([[[0, -1],[2, -1]], [[2, -1],[0, 1]], [[1, -1],[3, 2]],[[-1, 2],[1, 0]]])
        y = np.array([1, 1, 2, 2])
        Yf = np.array([[[0, -0.5],[1, -1]], [[0.2, -1],[0, 1]]])
        estimator = TRCAR(n_components=1, ensemble=True, n_jobs=-1)
        p_labels = estimator.fit(X, y, Yf)
        print(estimator.predict(np.array([[[0, -1.2],[0.5, -1]]])))
    """

    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_ = np.stack(
            [
                _trcar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCAR(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank TRCA-R algorithm (filter bank TRCA-R, fbTRCA-R) adds a filter bank analysis method to the TRCA-R
    algorithm, combining the fundamental and harmonic components of the signal. Multiple subband filters with different
    cutoff frequencies are utilized to filter the EEG signal to obtain the subband filtered signal. Subsequently,
    the correlation coefficients of the subband signals are summed according to a weighting function, and finally this
    weighted correlation coefficient sum is used as the feature discriminant[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Filter weights, defaults to None.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.


    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.
    Yf: ndarray
        Reference signal(n_classes, 2*n_harmonics, n_samples)


    References
    ----------
    .. [1] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed
       SSVEP-based brain-computer interface[J]. Journal of Neural Engineering, 2015, 12(4):046008.


    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using FBTRCAR

        import numpy as np
        from brainda.algorithms.decomposition import FBTRCAR
        X = np.zeros((4,22,22))
        for i in range(4):
        X[i,...] = np.identity(22)*0.3 + np.random.normal(-1,3,(22,22))*5
        y = np.array([1,1,2,2])
        Yf = X
        filterbank = [np.ones((3,6))]
        filterweights = np.array([[0.3, -0.1], [0.5, -0.1]])
        estimator = FBTRCAR(filterbank=filterbank,n_components=1,ensemble=True,filterweights=np.array(filterweights),n_jobs=-1)
        p_labels = estimator.fit(X, y, Yf)
        print(estimator.predict(np.identity(22)))

    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCAR(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


class SAxTRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The core idea of the SA-xTRCA algorithm is to iteratively optimize the phase differences
    between trials by calculating the correlation coefficients between each trial and the template,
    aiming to align trials with large phase discrepancies to achieve consistent phases.
    A simulated annealing (SA) algorithm is introduced to avoid the problem of local optima,
    which belongs to the supervised learning method[1]_.


    Parameters
    ----------
    n_components : int
        The number of spatial filters to retain after dimensionality reduction. Default is 1.

    ensemble : bool
        Whether to perform ensemble learning across categories. If True, spatial filters are learned separately
        for each class and then ensembled. Default is False.

    t0 : list
        Initial latency vector for each trial, representing the assumed starting point (in samples) of each trial.
        Typically initialized as a uniform list, e.g., [75, 75, ..., 75].

    tau : int
        The number of sampling points per trial (i.e., trial duration in samples). Default is 500 (2 seconds).

    tsearch0 : int
        The search range (in samples) for the sliding window when estimating latency. Default is 50.

    r : float
        The annealing coefficient used in the simulated annealing (SA) process to control temperature reduction.
        Default is 0.9.

    Q : int
        The maximum number of consecutive iterations allowed without improvement before forced termination
        of the optimization process. Default is 10.

    n_jobs : int or None
        Number of CPU cores to use for parallel processing. If None, only a single core is used.

    Attributes
    ----------

    T : float
    Initial temperature used in the simulated annealing process. Controls the starting level of randomness
    during the optimization. Default is 1000.

    T_min : float
        Minimum temperature for the simulated annealing process. When the temperature falls below this threshold,
        the iteration is terminated. Default is 1.

    t_i : list
        A list that stores the latency vectors (for all trials) recorded throughout the entire optimization process.

    t_update : list of lists
        A list where each element corresponds to a trial and contains the updated latency values of that trial
        during optimization. Initialized as a list of empty lists with length equal to the number of trials.

    fit_time_spend : list
        Records the time (in seconds or milliseconds, depending on implementation) consumed for fitting
        during each iteration of the simulated annealing process.

    D_change_len : list
        Stores the values of the optimization objective (e.g., lambda, correlation metric) across iterations.
        Useful for analyzing the convergence behavior of the algorithm.

    predict_time_spend : list
        Records the time taken to make predictions for each test sample or test round.

    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    temp : ndarray
        Individual average templates for each class, used for correlation comparison during testing.

    W : ndarray
        Spatial filters (projection matrices) computed for each class based on inter-trial correlation.

    References
    ----------
    .. [1] Wu J, He F, Xiao X, et al. SSVEP enhancement in mixed reality environment for
    brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2025, 420-430.


    """

    def __init__(
        self,
        n_components: int = 1,
        ensemble: bool = False,
        t0: list = [],
        tau: int = 500,
        tsearch0: int = 50,
        r: float = 0.9,
        Q: int = 10,
        n_jobs: Optional[int] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs
        self.t0 = t0  # Initial time vector
        self.tau = tau
        self.tsearch0 = tsearch0
        self.r = r
        self.T = 1000
        self.Q = Q
        self.T_min = 1
        self.t_i = []

    def trca_kernel(self, X: ndarray):
        """TRCA spatial filter calculate of SA-xTRCA.

        Parameters
        ----------
        X: ndarray
            Raw EEG data, shape(n_trials, n_channels, n_samples)

        Returns
        ----------
        W: ndarray
            Spatial filter, shape(n_channels, n_channels)
        D1: ndarray
            Feature vector, shape(n_channels,)
        Xb: ndarray
            Cablibrated EEG data, shape(n_trials, n_channels, n_samples)
        """
        N_trial = X.shape[0]
        N_chan = X.shape[1]

        X_bar = X / np.std(X, axis=-1, keepdims=True, ddof=1)

        X = X_bar
        Xb = np.zeros((N_trial, N_chan, self.tau))
        for trial_i in range(N_trial):
            Xb[trial_i] = X[
                trial_i, :, self.t[trial_i] : self.t[trial_i] + self.tau
            ] - np.mean(
                X[trial_i, :, self.t[trial_i] : self.t[trial_i] + self.tau],
                axis=-1,
                keepdims=True,
            )

        U = np.mean(Xb, axis=0)
        V = np.zeros((N_chan, N_chan))
        for trial_i in range(N_trial):
            V += (Xb[trial_i] @ Xb[trial_i].T) / N_trial
        S = N_trial / (N_trial - 1) / self.tau * (U @ U.T - V / N_trial)
        XQ = np.reshape(np.transpose(Xb, (1, 0, 2)), (N_chan, N_trial * self.tau))
        Q = XQ @ XQ.T / XQ.shape[1]

        [D, W] = eigh(S, Q)

        D1 = np.flip(np.sort(D))
        w_index = np.flip(np.argsort(D))
        W = W[:, w_index]
        y = W[:, 0].T @ XQ
        sgn = np.sign(np.squeeze(np.mean(XQ, axis=0)) @ y.T)
        W[:, 0] = sgn * W[:, 0]

        return W, D1, Xb

    def cross_relation(self, X: ndarray, W: ndarray, t: list = [], K: int = 0):
        """Cross-correlation coefficient calculate and latency vector update.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples)
        W: ndarray
            Spatial filter, shape(n_channels, n_channels)
        t: list
            latency vector, len(n_trials)
        K: int
            Index of trial latency to be updated

        Returns
        ----------

        t: ndarray
            latency vector, shape(n_trials,)
        """
        N_trial = X.shape[0]
        N_chan = X.shape[1]
        N_sample = X.shape[2]

        X_bar = np.reshape(np.transpose(X, (1, 0, 2)), (N_chan, N_trial * N_sample))
        y = W[:, 0].T @ X_bar
        y = y - np.mean(y, axis=0, keepdims=True)
        y = y / np.std(y, axis=0, keepdims=True, ddof=1)
        y = np.reshape(y, (N_trial, N_sample))

        tb = np.transpose(np.tile(t, (self.tau, 1)), (1, 0)) + np.tile(
            np.arange(0, self.tau), (len(t), 1)
        )
        yb = np.zeros((N_trial, self.tau))
        for trial_i in range(N_trial):
            yb[trial_i, :] = y[trial_i, tb[trial_i]]

        yb = yb - np.mean(yb, axis=1, keepdims=True)
        X_temp = np.mean(
            np.concatenate([yb[0:K], yb[K + 1 : N_trial]], axis=0),
            axis=0,
            keepdims=True,
        )
        _, t_new = self._cross_relation_kernel(X=X_temp, y2=y[K], t0=self.t0[K])

        self.t_update[K].append(t_new - t[K])
        t[K] = t_new
        return t

    def _cross_relation_kernel(self, X, y, t0):
        """Cross-correlation coefficient calculate and latency vector update.

        Parameters
        ----------
        X: ndarray
            Averaged EEG data, shape(1, n_channels, n_samples)
        y: ndarray
            Single EEG trial, shape(n_channels, n_samples)
        t0 : list
            Initial latency vector for each trial, len(n_trials)

        Returns
        ----------

        xcc_value: float
            cross-correlation coefficient
        t_updata: int
            The latency that maximizes the cross-correlation coefficient
        """
        X = np.squeeze(X)
        y = np.squeeze(y)
        tsearch = t0 + np.arange(
            int(-np.round(self.tsearch0 / 2)), int(np.round(self.tsearch0 / 2) + 1)
        )
        tb = np.transpose(np.tile(tsearch, (self.tau, 1)), (1, 0)) + np.tile(
            np.arange(0, self.tau), (tsearch.shape[0], 1)
        )
        ykb = y[tb]
        ykb = ykb - np.mean(ykb, axis=1, keepdims=True)
        xcc_value = ykb @ X.T / self.tau
        maxindex = np.argmax(xcc_value)
        t_updata = tsearch[maxindex]
        return xcc_value, t_updata

    def data_align(self, X_i: ndarray):
        """EEG data calibrate
        Parameters
        ----------
        X_i: ndarray
            Raw EEG data, shape(n_trials, n_channels, n_samples)

        Returns
        ----------
        X_align: ndarray
            Calibrated EEG data, shape(n_trials, n_channels, n_samples)
        """
        time_start = time.time()
        t = self.t0.copy()  # The time vector iterated during the iteration process
        self.t = t

        self.T = 1000
        self.T_min = 1

        N_trial = X_i.shape[0]
        N_sample = X_i.shape[2]
        N_chan = X_i.shape[1]

        self.t_update = [[] for i in range(N_trial)]

        # Iteration begins, first calculation
        _, D_0, _ = self.trca_kernel(X_i)
        q = 1
        D_old = D_0[0]

        D_change = []
        D_change.append(D_0[0])

        D_max = 0
        D = D_0

        while q < self.Q:
            D_old = D[0]

            for trial_i in range(N_trial):
                W, D, _ = self.trca_kernel(X_i)
                if D[0] >= D_max:
                    self.t_max = t.copy()
                    D_max = D[0]

                self.t = self.cross_relation(X_i, W, self.t, trial_i)
                W, D, _ = self.trca_kernel(X_i)
                D_change.append(D[0])

                # simulated annealing
                if D_change[-2] >= D[0] and self.T > self.T_min:
                    if math.exp((D_change[-2] - D[0]) / self.T) > np.random.uniform(
                        0,
                        1,
                        [
                            1,
                        ],
                    ):
                        L = len(self.t_update[trial_i])
                        if L == 1:
                            self.t_update[trial_i][-1] = int(
                                np.round(
                                    4
                                    * np.random.uniform(
                                        0,
                                        1,
                                        [
                                            1,
                                        ],
                                    )
                                    - 2
                                )
                            )
                            self.t[trial_i] += self.t_update[trial_i][-1]
                        else:
                            delta_t = (
                                np.mean(
                                    self.t_update[trial_i][
                                        slice(L - ((L <= 5) * L + (L > 5) * 5), L)
                                    ]
                                )
                                + 0.001
                            )
                            self.t[trial_i] = (
                                self.t[trial_i]
                                + int(np.sign(delta_t) * np.ceil(np.abs(delta_t)))
                                - int(self.t_update[trial_i][-1])
                            )
                            self.t_update[trial_i][-1] = int(
                                np.sign(delta_t) * np.ceil(np.abs(delta_t))
                            )
                        if self.t[trial_i] < 0:
                            self.t[trial_i] = 0
                        if (
                            self.t[trial_i]
                            > N_sample - self.tau - self.tsearch0 / 2 - 1
                        ):
                            self.t[trial_i] = N_sample - self.tau - 1

                        self.T *= self.r

            if (np.abs(D[0] - D_old)) / np.abs(D_old) < 10 ^ -4:
                break
            q += 1

        X_align = np.zeros((N_trial, N_chan, self.tau))
        for trial_i in range(N_trial):
            X_align[trial_i] = X_i[
                trial_i, :, int(self.t_max[trial_i]) : (self.t_max[trial_i] + self.tau)
            ]

        self.t_i.append(self.t)
        self.D_change = D_change
        self.D_change_len.append(len(D_change))
        del self.t

        time_end = time.time()
        self.fit_time_spend.append((time_end - time_start) * 1000)
        return X_align

    def fit(self, X: ndarray, y: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples)
        y: ndarray
            Labels, shape(n_trials,)
        """
        self.classes_ = np.unique(y)
        self.fit_time_spend = []
        self.D_change_len = []
        X_align = np.stack(
            [self.data_align(X[y == type_i]) for type_i in self.classes_]
        )
        X_align_array = []
        for i in range(X_align.shape[1]):
            X_align_array.append(X_align[:, i, :, :])
        X_align_array = np.concatenate(X_align_array, axis=0)

        W, temp = self.trca_train(X_align_array, y)
        self.W = W
        self.temp = temp
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        self.predict_time_spend = []
        X_temp = self.temp
        rhos = []
        for X_i in X:
            time_start = time.time()
            rou = []
            for type_i in range(self.classes_.shape[0]):
                if self.ensemble:
                    U = self.W[:, :, 0 : self.n_components]
                else:
                    U = self.W[type_i, :, 0 : self.n_components]

                xcc_value, t_new = self._cross_relation_kernel(
                    U.T @ X_temp[type_i], U.T @ X_i, self.t0[0]
                )
                X_test = X_i[:, t_new : t_new + self.tau]
                rou.append(
                    pearsonr(
                        np.squeeze(U.T @ X_temp[type_i]), np.squeeze(U.T @ X_test)
                    )[0]
                )
            rhos.append(rou)
            time_end = time.time()
            self.predict_time_spend.append((time_end - time_start) * 1000)

        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

    def trca_train(self, X: ndarray, y: ndarray):
        """TRCA spatial filter and templte calculate.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).

        Returns
        ----------
        W: ndarray
            Spatial filter, shape(self.classes_, n_channels, n_channels)
        temp: ndarray
            EEG templte, shape(self.classes_, n_channels, n_samples)
        """
        W = []
        temp = []
        for i in self.classes_:
            X_i = X[y == i]
            N_trial = X_i.shape[0]
            N_chan = X_i.shape[1]
            N_sample = X_i.shape[2]
            S = np.zeros((N_chan, N_chan))
            for trial_i in range(N_trial):
                for trial_j in range(N_trial):
                    x_i = X_i[trial_i]
                    x_j = X_i[trial_j]
                    S += x_i @ x_j.T
            X_bar = np.reshape(
                np.transpose(X_i, (1, 0, 2)), (N_chan, N_trial * N_sample)
            )
            X_bar = X_bar - np.mean(X_bar, axis=1, keepdims=True)

            Q = X_bar @ X_bar.T
            [D, w] = eigh(S, Q)
            index = np.flip(np.argsort(D))
            W.append(w[:, index])
            temp.append(np.mean(X_i, axis=0))
        W = np.stack(W, axis=0)
        temp = np.stack(temp, axis=0)
        return W, temp
