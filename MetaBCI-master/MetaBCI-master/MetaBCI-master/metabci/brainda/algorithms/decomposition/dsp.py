# -*- coding: utf-8 -*-
# DSP: Discriminal Spatial Patterns
# Authors: Swolf <swolfforever@gmail.com>
#          Junyang Wang <2144755928@qq.com>
# Last update date: 2022-8-11
# License: MIT License

from typing import Optional, List, Tuple
from itertools import combinations
import numpy as np
from scipy.linalg import eigh
from numpy import ndarray
from ..utils.covariance import nearestPD
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .base import robust_pattern
from .cca import FilterBankSSVEP


def xiang_dsp_kernel(
    X: ndarray, y: ndarray
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    DSP: Discriminal Spatial Patterns, only for two classes[1]_.
    Import train data to solve spatial filters with DSP,
    finds a projection matrix that maximize the between-class scatter matrix and
    minimize the within-class scatter matrix. Currently only support for two types of data.

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    X : ndarray
        EEG train data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels of EEG data, shape (n_trials, )

    Returns
    -------
    W : ndarray
        spatial filters, shape (n_channels, n_filters)
    D : ndarray
        eigenvalues in descending order
    M : ndarray
        mean value of all classes and trials, i.e. common mode signals, shape (n_channel, n_samples)
    A : ndarray
        spatial patterns, shape (n_channels, n_filters)

    Notes
    -----
    the implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(y == label) for label in labels])
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms, Ss = zip(
        *[
            (
                np.mean(X[y == label], axis=0),
                np.sum(
                    np.matmul(X[y == label], np.swapaxes(X[y == label], -1, -2)), axis=0
                ),
            )
            for label in labels
        ]
    )
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(
        Ss
        - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(
        n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )

    D, W = eigh(nearestPD(Sb), nearestPD(Sw))
    ix = np.argsort(D)[::-1]  # in descending order
    D, W = D[ix], W[:, ix]
    A = robust_pattern(W, Sb, W.T @ Sb @ W)

    return W, D, M, A


def xiang_dsp_feature(
    W: ndarray, M: ndarray, X: ndarray, n_components: int = 1
) -> ndarray:
    """
    Return DSP features in paper [1]_.

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    M : ndarray
        common template for all classes, shape (n_channel, n_samples)
    X : ndarray
        eeg test data, shape (n_trials, n_channels, n_samples)
    n_components : int, optional
        length of the spatial filters, first k components to use, by default 1

    Returns
    -------
    features: ndarray
        features, shape (n_trials, n_components, n_samples)

    Raises
    ------
    ValueError
        n_components should less than half of the number of channels

    Notes
    -----
    1. instead of meaning of filtered signals in paper [1]_., we directly return filtered signals.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    W, M, X = np.copy(W), np.copy(M), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_components].T, X - M)
    return features


class DSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    DSP: Discriminal Spatial Patterns

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    n_components : int
        length of the spatial filter, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    classes_ : int
        number of the EEG classes

    Attributes
    ----------
    n_components : int
        length of the spatial filter, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    classes_ : int
        number of the EEG classes
    W_ : ndarray, shape(n_channels, n_filters)
        Spatial filters, shape(n_channels, n_filters), in which n_channels = n_filters
    D_ : ndarray, shape(n_filters， )
        eigenvalues in descending order, shape(n_filters, )
    M_ : ndarray, shape(n_channels, n_samples)
        mean value of all classes and trials, i.e. common mode signals, shape(n_channels, n_samples)
    A_ : ndarray, shape(n_channels, n_filters)
        spatial patterns, shape(n_channels, n_filters)
    templates_: ndarray, shape(n_classes, n_filters, n_samples)
        templates of train data, shape(n_classes, n_filters, n_samples)

    """

    def __init__(self, n_components: int = 1, transform_method: str = "corr"):
        self.n_components = n_components
        self.transform_method = transform_method

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """
        Import the train data to get a model.

        Parameters
        ----------
        X : ndarray
            train data, shape(n_trials, n_channels, n_samples)
        y : ndarray
            labels of train data, shape (n_trials, )
        Yf : ndarray
            optional parameter

        Returns
        -------
        W_ : ndarray
            spatial filters, shape (n_channels, n_filters), in which n_channels = n_filters
        D_ : ndarray
            eigenvalues in descending order, shape (n_filters, )
        M_ : ndarray
            template for all classes, shape (n_channel, n_samples)
        A_ : ndarray
            spatial patterns, shape (n_channels, n_filters)
        templates_ : ndarray
            templates of train data, shape (n_channels, n_filters, n_samples)
        """
        X -= np.mean(X, axis=-1, keepdims=True)
        self.classes_ = np.unique(y)
        self.W_, self.D_, self.M_, self.A_ = xiang_dsp_kernel(X, y)

        self.templates_ = np.stack(
            [
                np.mean(
                    xiang_dsp_feature(
                        self.W_, self.M_, X[y == label], n_components=self.W_.shape[1]
                    ),
                    axis=0,
                )
                for label in self.classes_
            ]
        )
        return self

    def transform(self, X: ndarray):
        """
        Import the test data to get features.

        Parameters
        ----------
         X : ndarray
            test data, shape(n_trials, n_channels, n_samples)

        Returns
        -------
        feature : ndarray, shape(n_trials,n_classes)
            correlation coefficients of templates of train data and features of test data, shape(n_trials, n_classes)
        """
        n_components = self.n_components
        X -= np.mean(X, axis=-1, keepdims=True)
        features = xiang_dsp_feature(self.W_, self.M_, X, n_components=n_components)
        if self.transform_method is None:
            return features.reshape((features.shape[0], -1))
        elif self.transform_method == "mean":
            return np.mean(features, axis=-1)
        elif self.transform_method == "corr":
            return self._pearson_features(
                features, self.templates_[:, :n_components, :]
            )
        else:
            raise ValueError("non-supported transform method")

    def _pearson_features(self, X: ndarray, templates: ndarray):
        """
        Calculate pearson correlation coefficient.

        Parameters
        ----------
        X : ndarray
            features of test data after spatial filters, shape(n_trials, n_components, n_samples)
        templates : ndarray
            templates of train data, shape(n_classes, n_components, n_samples)

        Returns
        -------
        corr : ndarray
            pearson correlation coefficient, shape(n_trials, n_classes)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        templates = np.reshape(templates, (-1, *templates.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = templates - np.mean(templates, axis=-1, keepdims=True)
        X = np.reshape(X, (X.shape[0], -1))
        templates = np.reshape(templates, (templates.shape[0], -1))
        istd_X = 1 / np.std(X, axis=-1, keepdims=True)
        istd_templates = 1 / np.std(templates, axis=-1, keepdims=True)
        corr = (X @ templates.T) / (templates.shape[1] - 1)
        corr = istd_X * corr * istd_templates.T
        return corr

    def predict(self, X: ndarray):
        """
        Import the templates and the test data to get prediction labels.

        Parameters
        ----------
        X : ndarray
            test data, shape(n_trials, n_channels, n_samples)

        Returns
        -------
        labels : ndarray
            prediction labels of test data, shape(n_trials,)
        """
        feat = self.transform(X)
        if self.transform_method == "corr":
            labels = self.classes_[np.argmax(feat, axis=-1)]
        else:
            raise NotImplementedError()
        return labels


class FBDSP(FilterBankSSVEP, ClassifierMixin):
    """
    FBDSP: FilterBank DSP

    Author: Swolf <swolfforever@gmail.com>

    Created on: 2021-1-07

    Update log:

    Parameters
    ----------
    filterbank : list
        bandpass filterbank, ([float, float],...)
    n_components : int
        length of the spatial filters, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    filterweights : ndarray
        filter weights, optional parameter, by default None
    n_jobs : int
        optional parameter, by default None

    Attributes
    ----------
    filterbank : list[[float, float], …]
        bandpass filterbank, ([float, float],...)
    n_components : int
        length of the spatial filters, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    filterweights : ndarray
        filter weights, optional parameter, by default None
    n_jobs : int
        optional parameter, by default None
    classes_ : int
        number of classes
    W_ : ndarray, shape(n_channels, n_filters)
        spatial filter, shpe(n_channels, n_filters), in which n_channels = n_filters
    D_ : ndarray, shape(n_filters, )
        eigenvalues in descending order, shape(n_filters, )
    M_ : ndarray, shape(n_channels, n_samples)
        mean value of all classes and trials, i.e. common mode signals, shape(n_channels, n_samples)
    A_ : ndarray, shape(n_channels, n_filters)
        spatial patterns, shape(n_channels, n_filters)
    templates_ : ndarray, shape(n_classes, n_filters, n_samples)
        templates of train data, shape(n_classes, n_filters, n_samples)
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        transform_method: str = "corr",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.transform_method = transform_method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            DSP(n_components=n_components, transform_method=transform_method),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """
        Import the test data to get features.

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            train data, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape(n_trials, )
            labels of train data, shape (n_trials, )
        Yf : ndarray
            optional parameter,

        Returns
        -------
        W_ : ndarray
            spatial filters, shape (n_channels, n_filters)
        D_ : ndarray
            eigenvalues in descending order
        M_ : ndarray
            template for all classes, shape (n_channel, n_samples)
        A_ : ndarray
            spatial patterns, shape (n_channels, n_filters)
        templates_ : ndarray
            templates of train data, shape (n_channels, n_filters, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """
        Import the templates and the test data to get prediction labels.

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            test data, shape(n_trials, n_channels, n_samples)

        Returns
        -------
        labels : ndarray, shape(n_trials, )
            prediction labels of test data, shape(n_trials, )

        See Also
        -------
        FilterBankSSVEP : filterbank analysis (base)
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


class DCPM(DSP, ClassifierMixin):
    """
    DCPM: discriminative canonical pattern matching [1]_.

    Author: Junyang Wang <2144755928@qq.com>

    Create on: 2022-6-26

    Update log:

    Parameters
    ----------
    n_components : int
        length of the spatial filters, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    n_rpts : int
        repetition times in a block

    Attributes
    ----------
    n_components : int
        length of the spatial filters, first k components to use, by default 1
    transform_method : str
        method of template matching, by default ’corr‘ (pearson correlation coefficient)
    n_rpts : int
        repetition times in a block
    classes_ : int
        number of classes
    combinations_ : list, ([int, int], …)
        combinations of two classes in all classes
    n_combinations : int
        numbers of combinations
    Ws : ndarray, shape(n_channels, n_components *n_combinations)
        spatial filter, shpe(n_channels, n_components * n_combinations)
    templates : ndarray, shape(n_classes, n_components*n_combinations, n_samples)
        templates of train data, shape(n_classes, n_components * n_combinations, n_samples)
    M : ndarray, shape(n_channels, n_samples)
        mean value of all classes and trials, i.e. common mode signals, shape(n_channels, n_samples)

    References
    ----------
    .. [1]	Xu MP, Xiao XL, Wang YJ, et al. A brain-computer interface based on miniature-event-related
        potentials induced by very small lateral visual stimuli[J]. IEEE Transactions on Biomedical
        Engineering, 2018:65(5), 1166-1175.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: An example using DCPM

       from brainda.algorithms.decomposition.dsp import DCPM
       X = np.array(data.get(‘X’))     #data(n_trials, n_channels, n_times)
       y = data.get(‘Y’)               #labels(n_trials)
       estimator = DCPM(n_components=2,transform_method=’corr’, n_rpts=1)
       accs = []
       # use ‘fit’ to get the model of train data;
       # use ‘predict’ to get the prediction labels of test data;
       p_labels=estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
       accs.append(np.mean(p_labels==y[test_ind]))
       print(np.mean(accs))


    See Also
    ----------
    pearson_features: calculate pearson correlation coefficients
    """

    def __init__(
        self, n_components: int = 1, transform_method: str = "corr"
    ):
        self.n_components = n_components
        self.transform_method = transform_method

        super().__init__(n_components=n_components, transform_method=transform_method)

    def fit(self, X: ndarray, y: ndarray):  # type: ignore[override]
        """
        Import the train data to get a model: Ws, templates, M.

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            train data, shape(n_trials, n_channels, n_samples)
        y : ndarray, shape(n_trials, )
            labels of train data, shape(n_trials, )

        Returns
        -------
        Ws : ndarray
            spatial filters of train data, shape(n_channels, n_components * n_combinations)
        templates : ndarray
            templates of train data, shape(n_classes, n_components*n_combinations, n_samples)
        M : ndarray
            mean of train data (common-mode signals), shape(n_channels, n_samples)
        """
        X -= np.mean(X, axis=-1, keepdims=True)
        X /= np.std(X, axis=(-1, -2), keepdims=True)  # standardize the train data
        self.classes_ = np.unique(y)  # number of the eeg classes
        self.combinations_ = list(
            combinations(range(self.classes_.shape[0]), 2)
        )  # combine two classes in all classes: C(2,n_classes)
        self.n_combinations = len(self.combinations_)
        # get the W(spatial filter) for each combination
        Ws = []
        for icomb, comb in enumerate(self.combinations_):
            Xs_train = np.concatenate(
                [X[y == self.classes_[comb[i]]] for i in range(2)], axis=0
            )  # data of two classes
            ys_train = np.concatenate(
                [y[y == self.classes_[comb[i]]] for i in range(2)], axis=0
            )  # labels of two classes
            W, _, _, _ = xiang_dsp_kernel(
                Xs_train, ys_train
            )  # length of W is n_components
            Ws.append(W[:, : self.n_components])  # W(n_channels, n_components)
        # concatenate W of each combination
        self.Ws = np.concatenate(
            Ws, axis=-1
        )  # Ws(n_channels, n_components * n_combinations)
        # mean of same class
        T = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_], axis=0
        )  # T(n_classes, n_channels, n_samples)
        # mean of all classes and trials
        self.M = np.mean(T, axis=0)  # M(n_channels, n_samples)
        # get the templates of train data
        self.templates = np.matmul(
            self.Ws.T, T - self.M
        )  # templates(n_classes, n_components*n_combinations, n_samples)
        return self

    def transform(self, X: ndarray):
        """
        Import the test data to get features.

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            test data, shape(n_trials, n_channels, n_samples)

        Returns
        -------
        feature : ndarray, shape(n_trials,n_classes)
            features of test data, shape(n_trials, n_classes)
        """
        X -= np.mean(X, axis=-1, keepdims=True)
        X /= np.std(X, axis=(-1, -2), keepdims=True)  # standardize train data
        Ws = self.Ws
        M = self.M
        X_feature = np.matmul(
            Ws.T, X - M
        )  # X_feature(n_trials, n_components*n_combinations, n_samples)
        feature = self._pearson_features(
            X_feature, self.templates
        )  # feature(n_trials, n_classes)
        return feature

    def predict(self, X: ndarray):
        """
        Import the templates and the test data to get prediction labels.

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            test data, shape(n_trials, n_channels, n_samples)

        Returns
        -------
        labels : ndarray, shape(n_trials, )
            prediction labels of test data, shape(n_trials, )
        """
        feat = self.transform(X)
        labels = np.argmax(feat, axis=-1)  # prediction labels()
        labels = np.concatenate(
            [self.classes_[self.classes_ == self.classes_[labels[i]]] for i in range(labels.shape[0])], axis=0
        )
        return labels


# pearson correlation coefficient
def pearson_features(X, templates):
    '''
    Calculate pearson correlation coefficient.

    Parameters
    ----------
    X : ndarray
        features of test data after spatial filters, shape(n_trials, n_components, n_samples)
    templates : ndarray
        templates of train data, shape(n_classes, n_components, n_samples)

    Returns
    -------
    corr : ndarray
        pearson correlation coefficient, shape(n_trials, n_classes)
    '''

    X = np.reshape(X, (-1, *X.shape[-2:]))
    templates = np.reshape(templates, (-1, *templates.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    templates = templates - np.mean(templates, axis=-1, keepdims=True)
    X = np.reshape(X, (X.shape[0], -1))
    templates = np.reshape(templates, (templates.shape[0], -1))
    istd_X = 1 / np.std(X, axis=-1, keepdims=True)
    istd_templates = 1 / np.std(templates, axis=-1, keepdims=True)
    corr = (X @ templates.T) / (templates.shape[1] - 1)
    corr = istd_X * corr * istd_templates.T
    return corr
