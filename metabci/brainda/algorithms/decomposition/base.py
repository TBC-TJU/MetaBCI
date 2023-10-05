# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
from typing import Optional, List, Tuple, Union
import warnings
import numpy as np
from numpy import ndarray
from scipy.linalg import solve
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone
from metabci.brainda.datasets.base import BaseTimeEncodingDataset
import mne


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
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging."
           Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A


class FilterBank(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        filterbank: List[ndarray],
        n_jobs: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.filterbank = filterbank
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: Optional[ndarray] = None, **kwargs):
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(self.filterbank))
        ]
        X = self.transform_filterbank(X)
        for i, est in enumerate(self.estimators_):
            est.fit(X[i], y, **kwargs)
        # def wrapper(est, X, y, kwargs):
        #     est.fit(X, y, **kwargs)
        #     return est
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], y, kwargs) for i, est in enumerate(self.estimators_))
        return self

    def transform(self, X: ndarray, **kwargs):
        X = self.transform_filterbank(X)
        feat = [est.transform(X[i], **kwargs) for i, est in enumerate(self.estimators_)]
        # def wrapper(est, X, kwargs):
        #     retval = est.transform(X, **kwargs)
        #     return retval
        # feat = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], kwargs) for i, est in enumerate(self.estimators_))
        feat = np.concatenate(feat, axis=-1)
        return feat

    def transform_filterbank(self, X: ndarray):
        Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in self.filterbank])
        return Xs


class FilterBankSSVEP(FilterBank):
    """Filter bank analysis for SSVEP."""

    def __init__(
        self,
        filterbank: List[ndarray],
        base_estimator: BaseEstimator,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.filterweights = filterweights
        super().__init__(base_estimator, filterbank, n_jobs=n_jobs)

    def transform(self, X: ndarray):  # type: ignore[override]
        features = super().transform(X)
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            return np.sum(
                features * self.filterweights[np.newaxis, :, np.newaxis], axis=1
            )


class TimeDecodeTool:
    def __init__(self, dataset: BaseTimeEncodingDataset, feature_operation: str = 'sum'):
        # Get minor event from the dataset
        minor_events = dataset.minor_events
        minor_class = list()
        for event in minor_events.values():
            minor_class.append(event[0])
        minor_class.sort()
        self.minor_class = np.array(minor_class)
        self.encode_map = dataset.encode
        self.encode_loop = dataset.encode_loop
        self.feature_operation = feature_operation

    def _trial_feature_split(self, key: str, feature: ndarray):
        key_encode = self.encode_map[key]
        key_encode_len = len(key_encode)
        if key_encode_len * self.encode_loop != feature.shape[0]:
            raise ValueError('Epochs in the test trial does not same '
                             'as the presetting parameter in dataset')
        # create a space for storage feature
        feature_storage = np.zeros((self.encode_loop, key_encode_len, *feature.shape[1:]))
        for row in range(self.encode_loop):
            for col in range(key_encode_len):
                feature_storage[row][col] = feature[row * key_encode_len + col, :]

        return key, feature_storage

    def _features_operation(self, feature_storage: ndarray, fold_num=6):
        if fold_num > np.shape(feature_storage)[0]:
            raise ValueError("The number of trial stacks cannot exceeds %d" % np.shape(feature_storage)[0])
        if self.feature_operation == 'sum':
            sum_feature = np.sum(feature_storage[0:fold_num], axis=0, keepdims=False)
            return sum_feature

    def _predict(self, features: ndarray):
        predict_labels = self.minor_class[np.argmax(features, axis=-1)]
        return predict_labels

    def _predict_p300(self, features: ndarray):
        code_len = features.shape[0]
        half_len = int(code_len/2)
        predict_row = np.argmax(features[:half_len, -1])
        predict_col = np.argmax(features[half_len:, -1])+6
        predict_labels = np.ones_like(self.minor_class, dtype=int)
        predict_labels[predict_row] = 2
        predict_labels[predict_col] = 2
        return predict_labels

    def _find_command(self, predict_labels: ndarray):
        for key, value in self.encode_map.items():
            if np.array_equal(np.array(value), predict_labels):
                return key
        return None

    def decode(self, key: str, feature: ndarray, fold_num=6, paradigm='avep'):
        if feature.ndim < 2:
            feature = feature[:, np.newaxis]
        alpha_key, feature_storage = self._trial_feature_split(key, feature)
        merge_features = self._features_operation(feature_storage, fold_num)
        predict_labels = []
        if paradigm == 'avep':
            predict_labels = self._predict(merge_features)
        elif paradigm == 'p300':
            predict_labels = self._predict_p300(merge_features)
        command = self._find_command(np.array(predict_labels))
        return command

    def target_calibrate(self, y, key):
        y_tar = []
        for i in range(len(y)):
            character = key.values[i]
            target_id = np.where(
                np.array(self.encode_map[character]) == 2)[0]+1
            target_loc = []
            event = y[i].copy()
            for j in target_id:
                target_loc = np.append(target_loc, np.where(event == j))
            target_loc = np.array(target_loc, dtype=int)

            event[:] = 1
            event[target_loc] = 2
            y_tar.append(event)
        return y_tar

    def resample(self, x, fs_old, fs_new, axis=None):
        if axis is None:
            axis = x.ndim-1
        down_factor = fs_old/fs_new
        x_1 = mne.filter.resample(x, down=down_factor, axis=axis)
        return x_1

    def epoch_sort(self, X, y):
        code_len = len(self.minor_class)
        X_sort = [[] for i in range(len(X))]
        Y_sort = [[] for i in range(len(y))]
        for char_i in range(len(X)):
            for loop_i in range(self.encode_loop):
                epoch_id = np.arange(loop_i*code_len, (loop_i+1)*code_len)
                y_i = y[char_i][epoch_id]
                x_i = X[char_i][epoch_id]

                id = np.argsort(y_i)
                x_sort = x_i[id, :, :]
                y_sort = y_i[id]
                X_sort[char_i].append(x_sort)
                Y_sort[char_i].append(y_sort)
            X_sort[char_i] = np.concatenate(X_sort[char_i], axis=0)
            Y_sort[char_i] = np.concatenate(Y_sort[char_i], axis=0)
        return X_sort, Y_sort


def generate_filterbank(
    passbands: List[Tuple[float, float]],
    stopbands: List[Tuple[float, float]],
    srate: int,
    order: Optional[int] = None,
    rp: float = 0.5,
):
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)

        filterbank.append(sos)
    return filterbank


def generate_cca_references(
    freqs: Union[ndarray, int, float],
    srate,
    T,
    phases: Optional[Union[ndarray, int, float]] = None,
    n_harmonics: int = 1,
):
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = np.array([freqs])
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = np.array([phases])
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T * srate))

    Yf = []
    for i in range(n_harmonics):
        Yf.append(
            np.stack(
                [
                    np.sin(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                    np.cos(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                ],
                axis=1,
            )
        )
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
        total_proj = np.sum(u * s, axis=0)
        signs = np.sign(total_proj)

        random_idx = signs == 0
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn(
                "The magnitude is close to zero, the sign will become arbitrary."
            )

        u = u * signs

        return u, s
    else:
        left_proj = np.sum(s[:, np.newaxis] * vh, axis=-1)
        right_proj = np.sum(u * s, axis=0)
        total_proj = left_proj + right_proj
        signs = np.sign(total_proj)

        random_idx = signs == 0
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn(
                "The magnitude is close to zero, the sign will become arbitrary."
            )

        u = u * signs
        vh = signs[:, np.newaxis] * vh

        return u, s, vh
