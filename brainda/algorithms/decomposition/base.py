# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
from typing import Optional, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.linalg import solve
from scipy.signal import sosfiltfilt, cheby2, cheb2ord, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone

def robust_pattern(W: ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_channels, n_filters).
    Cx : ndarray
        Covariance matrix of eeg data, shape (n_channels, n_channels).
    Cs : ndarray
        Covariance matrix of source data, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_channels, n_patterns), each column is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A   

class FilterBank(BaseEstimator, TransformerMixin):
    def __init__(self, base_estimator: Optional[BaseEstimator] = None,
            filterbank: Optional[List[ndarray]] = None):
        self.base_estimator = base_estimator
        self.filterbank = filterbank

    def fit(self, X: ndarray, y: ndarray):
        # transform filterbank
        X = self.transform_filterbank(X)
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(X))]
        for i, estimator in enumerate(self.estimators_):
            estimator.fit(X[i], y)
        return self

    def transform(self, X: ndarray):
        X = self.transform_filterbank(X)
        features = np.concatenate(
            [est.transform(X[i]) for i, est in enumerate(self.estimators_)], axis=-1)
        return features

    def _check_filterbank(self):
        if hasattr(self, 'filterbank') and isinstance(self.filterbank, list):
            if self.filterbank[0].ndim != 2 or self.filterbank[0].shape[1] != 6:
                raise ValueError("only sos coefficients supported.")
            return True
        return False

    def transform_filterbank(self, X: ndarray):
        if self._check_filterbank():
            Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in self.filterbank])
            return Xs
        else:
            return X[np.newaxis, ...]

def generate_filterbank(
        passbands: List[Tuple[float, float]],
        stopbands: List[Tuple[float, float]],
        srate: int):
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        # N, wn = cheb2ord(wp, ws, 3, 40, fs=srate)
        # sos = cheby2(N, 0.5, wn, btype='bandpass', output='sos', fs=srate)
        N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
        sos = cheby1(N, 0.5, wn, btype='bandpass', output='sos', fs=srate)

        filterbank.append(sos)
    return filterbank

def generate_cca_references(freqs, srate, T, 
        phases: Optional[ndarray] = None,
        n_harmonics: int = 1):
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = [freqs] 
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = [phases] 
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T*srate))

    Yf = []
    for i in range(n_harmonics):
        Yf.append(np.stack([
            np.sin(2*np.pi*(i+1)*freqs*t + np.pi*phases),
            np.cos(2*np.pi*(i+1)*freqs*t + np.pi*phases)], axis=1))
    Yf = np.concatenate(Yf, axis=1)
    return Yf

def sign_flip(u, s, vh=None):
    """Flip signs of SVD or EIG using the method in paper [1]_.

    Parameters
    ----------
    u: ndarray
        left singular vectors, shape (M, K).
    s: ndarray
        singular values, shape (K,).
    vh: ndarray or None
        transpose of right singular vectors, shape (K, N).

    Returns
    -------
    u: ndarray
        corrected left singular vectors.
    s: ndarray
        singular values.
    vh: ndarray
        transpose of corrected right singular vectors.

    References
    ----------
    .. [1] https://www.sandia.gov/~tgkolda/pubs/pubfiles/SAND2007-6422.pdf
    """
    if vh is None:
        total_proj = np.sum(u*s, axis=0)
        signs = np.sign(total_proj)
        
        random_idx = (signs==0)
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn("The magnitude is close to zero, the sign will become arbitrary.")
            
        u = u*signs
        
        return u, s
    else:
        left_proj = np.sum(s[:, np.newaxis]*vh, axis=-1)
        right_proj = np.sum(u*s, axis=0)
        total_proj = left_proj + right_proj
        signs = np.sign(total_proj)
        
        random_idx = (signs==0)
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn("The magnitude is close to zero, the sign will become arbitrary.")

        u = u*signs
        vh = signs[:, np.newaxis]*vh

        return u, s, vh


        
