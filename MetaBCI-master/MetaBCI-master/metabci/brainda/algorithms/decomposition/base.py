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


def robust_pattern(W : ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.
        Referring to the method mentioned in article [1],the constructed spatial filter only shows how to combine
        information from different channels to extract signals of interest from EEG signals, but if our goal is
        neurophysiological interpretation or visualization of weights, activation patterns need to be constructed
        from the obtained spatial filters.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

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
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging.
           Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A


class FilterBank(BaseEstimator, TransformerMixin):
    """
    Filter bank decomposition is a bandpass filter array that divides the input signal into
    multiple subband components and obtains the eigenvalues of each subband component.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    base_estimator : class
        Estimator for model training and feature extraction.
    filterbank : list[ndarray]
        A bandpass filter bank used to divide the input signal into multiple subband components.
    n_jobs : int
        Sets the number of CPU working cores. The default is None.

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain-computer interface[J].
    Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.
    """
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
        """
        Training model

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : None
            Training signal (parameters can be ignored, only used to maintain code structure).
        y : None
            Label data (ibid., ignorable).
        Yf : None
            Reference signal (ibid., ignorable).
        """
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
        """
        The parameters stored in self are used to convert X into features, and X is filtered through the filter bank to
        obtain the eigenvalues of each subband component.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Test the signal.

        Returns
        -------
        feat : ndarray, shape(n_trials, n_fre)
            Feature array.
        """
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
        """
        The input signal is filtered through a filter bank.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Input signal.

        Returns
        -------
        Xs: ndarray, shape(Nfb, n_trials, n_channels, n_samples)
            Individual subband components of the input signal.
        """
        Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in self.filterbank])
        return Xs


class FilterBankSSVEP(FilterBank):
    """
    Filter bank analysis for SSVEP.
    The SSVEP is analyzed using filter banks, that is, multiple filters are combined to decompose the SSVEP signal
    into specific segments (subbands containing the original data) and obtain its characteristic data.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    filterbank : list[ndarray]
        The filter bank.
    base_estimator : class
        Estimator for model training and feature extraction.
    filterweights : ndarray
        Filter weight, default is None.
    n_jobs : int
        Sets the number of CPU working cores. The default is None.
    """

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
        """
        X is converted into features by using the parameters stored in self, and the eigenvalues of each subband
        component are obtained after the input signal is filtered by the filter bank.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Test the signal.

        Returns
        -------
        features : ndarray, shape(n_trials, n_fre)
            Feature array.
        """
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
    """
    Decoding tool set for TDMA coding paradigm. Applicable data sets include P300 speller data set and aVEP speller
    data.The main functions include: dividing the trial according to the minor event, downsampling the data,
    and determining the target character (or instruction) according to the judgment result of the trial.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    dataset : BaseTimeEncodingDataset
        The data set to be decoded.
    feature_operation : str
        An operation performed after feature extraction for each attempt of the same class.
    """
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
        """
        The extracted feature is divided according to the character big tag (key, which is used to determine the length
        of the encoding sequence, which can be any big tag), the stimulus repetition cycle (self.encode_loop) and the
        encoding sequence corresponding to the big tag (self.encode_map).

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters ---------- key : str Character large label. feature : ndarray, shape(n_trials, n_class) A
        multidimensional array of the features of multiple attempts. The size of the array is the number of attempts
        x the number of template categories. Where the number of attempts is equal to the number of stimulus repeats
        * the length of the encoding sequence (key_encode_len).

        Returns
        -------
        key : str
            Character large label.
        feature_storage : ndarray, shape(encode_loop, key_encode_len, n_class)
            A multi-dimensional array of the features of multiple attempts after partitioning. The size of the array is
            the number of rounds * the length of the encoding sequence * the number of template classes
        """
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
        """
        The feature stack and other operations are carried out on the feature array with multiple repetitions.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        feature_storage : ndarray, shape(encode_loop, key_encode_len, n_samples)
            A multi-dimensional array composed of the features of multiple attempts after partitioning. The size of the
            array is the number of rounds * the length of the encoding sequence * the length of the feature vector.
        fold_num : int
            The stimulation was repeated.

        Returns
        -------
        sum_feature : ndarray, shape(key_encode_len, n_class)
            A multi-dimensional array composed of features of multiple attempts after superposition. The size of the
            array is the length of the encoding sequence * category.
        """
        if fold_num > np.shape(feature_storage)[0]:
            raise ValueError("The number of trial stacks cannot exceeds %d" % np.shape(feature_storage)[0])
        if self.feature_operation == 'sum':
            sum_feature = np.sum(feature_storage[0:fold_num], axis=0, keepdims=False)
            return sum_feature

    def _predict(self, features: ndarray):
        """
        To predict the category of trials based on the characteristics of the trials, the applicable data set includes
        aVEP speller data.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        features : ndarray, shape(key_encode_len, n_class)
            The eigenvalues are computed from multiple attempts and different templates

        Returns
        -------
        predict_labels : ndarray, shape(key_encode_len, 1)
            The class of multiple attempts predicted from the eigenvalue.
        """
        predict_labels = self.minor_class[np.argmax(features, axis=-1)]
        return predict_labels

    def _predict_p300(self, features: ndarray):
        """
        The decoding method specifically designed for the classical column P300 speller can predict the category of the
        trial according to the characteristics of the trial.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        features : ndarray, shape(key_encode_len, n_class)
            The eigenvalues are computed from multiple attempts and different templates.

        Returns
        -------
        predict_labels : ndarray, shape(key_encode_len, 1)
            The class of multiple attempts predicted from the eigenvalue.
        """
        code_len = features.shape[0]
        half_len = int(code_len/2)
        predict_row = np.argmax(features[:half_len, -1])
        predict_col = np.argmax(features[half_len:, -1])+6
        predict_labels = np.ones_like(self.minor_class, dtype=int)
        predict_labels[predict_row] = 2
        predict_labels[predict_col] = 2
        return predict_labels

    def _find_command(self, predict_labels: ndarray):
        """
        The class of the character to be tested is determined by comparing the encoding sequence of each
        character (instruction) with the class predicted from multiple trials.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        predict_labels : ndarray, shape(key_encode_len, 1)
            The class of multiple attempts predicted from the eigenvalue.

        Returns
        -------
        key or none : str or none
            The character to be tested is predicted according to the class sequence of the time to be tested. If the
            predicted sequence exists in the encoded sequence of the data set, the character corresponding to the
            predicted sequence is output; If the prediction sequence does not exist in the dataset encoding sequence,
            output none.
        """
        for key, value in self.encode_map.items():
            if np.array_equal(np.array(value), predict_labels):
                return key
        return None

    def decode(self, key: str, feature: ndarray, fold_num=6, paradigm='avep'):
        """
        The data is decoded according to character large label (used to determine the encoding sequence length, which
        can be any large label) characteristics, stimulus repetition cycles (fold_num), and normal form types.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        key : str
            Character large label.
        feature : ndarray, shape(n_trials, n_class)
            A multidimensional array of the features of multiple attempts. The size of the array is the number of
            attempts x the number of template categories. Where the number of attempts is equal to the number of
            stimulus repeats * the length of the encoding sequence (key_encode_len).
        fold_num : int
            The stimulation was repeated.
        paradigm : str
            Type of paradigm.

        Returns
        -------
        command : str
            The character to be tested is predicted according to the class sequence of the test.
        """
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
        """
        A trial identification method specifically designed for the classic column P300 speller. According to the trial
        label (y) and character label (key) of the labeled column in the P300 data set, the trial label is converted
        into a small label that can label "target" and "non-target".

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        y : list
            Each element is a character corresponding to all the try labels.
        key:
            A large label, which contains the label value (key.index) and the character corresponding to the label
            (key.value).

        Returns
        -------
        y_tar : list
            Each element is all the small labels corresponding to a character (labeled "target" and "non-target").
        """
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
        """
        Each element is all the small labels that correspond to a character (labeled "target" and "non-target").

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        x : ndarray
            Each element is a character corresponding to all the try labels.
        fs_old : float
            The original sampling rate of x.
        fs_new : float
            Sampling rate of resampling.
        axis:
            Dimensions of resampling.

        Returns
        -------
        x_1 : ndarray
            Data after resampling.
        """
        if axis is None:
            axis = x.ndim-1
        down_factor = fs_old/fs_new
        x_1 = mne.filter.resample(x, down=down_factor, axis=axis)
        return x_1

    def epoch_sort(self, X, y):
        """
        A trial-ordering method designed specifically for the classic column P300 speller.
        The trials are sorted in ascending order according to the trial label of a single round of characters.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : list
            Pre-sort data for multiple characters, where each element represents the data for all attempts of a
            character.
        y : list
            A multi-character trial tag, where each element represents the label value of all the tries of a character,
            and the label value represents the currently blinking row or column.

        Returns
        -------
        X_sort : list
            The sorted data of multiple characters is arranged in ascending order of the label value, where each element
            represents the data of all attempts of a character.
        Y_sort : list
            After the sorting of multiple characters, each element in the ascending order of the label value represents
            the label value of all the tries of a character. The label value represents the current blinking row or
            column.
        """
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
    """
    Create a filter bank, that is, obtain a bandpass filter coefficient that can divide the input signal into multiple
    subband components.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    passbands : list or tuple(float, float)
        Passband parameters.
    stopbands : list or tuple(float, float)
        Stopband parameters.
    srate : float
        Sampling rate.
    order : int
        Filter order.
    rp : float
        The maximum ripple allowed in the passband below the unit gain is 0.5 by default.

    Returns
    -------
    Filterbank：ndarray, shape(len(passbands), N, 6)
        Filter bank coefficient.
    """
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
    """
    Construct a sine-cosine reference signal for canonical correlation analysis (CCA).

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    freqs : int or float
        Frequency.
    srate : int
        Sampling rate.
    T : int
        Sampling time.
    phases : int or float
        Phase, default is None.
    n_harmonics : int
        The number of harmonics. The default value is 1.

    Returns
    -------
    Yf：ndarray, shape(srate*T, n_harmonics*2)
        Sine and cosine reference signal.
    """
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

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

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
