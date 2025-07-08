# -*- coding: utf-8 -*-
# BCI Performance Evaluation
# Authors: Ruixinluo <ruixin_luo@gtju.edu.cn>
# Date: 2023/1/07
# License: MIT License
import cProfile
import io
import pstats
import typing
from pstats import SortKey
from typing import Any

from numpy import ndarray
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper


def _accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """Accuracy classification score

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    acc: float
        accuracy classification score.
    """

    if y_true.size != y_pred.size:
        raise ValueError(
            """The size of the predicted label and the real label should be the same""")
    acc = metrics.accuracy_score(y_true, y_pred)

    return acc


def _balance_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """Compute the balanced accuracy to deal with imbalanced datasets.
       It is defined as the average of recall obtained on each class.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    acc: float
        balanced accuracy score.
    """

    if y_true.size != y_pred.size:
        raise ValueError(
            """The size of the predicted label and the real label should be the same""")
    acc = metrics.balanced_accuracy_score(y_true, y_pred)

    return acc


def _theoretical_itr(y_true: ndarray, y_pred: ndarray, Tw: float) -> float:
    """Theoretical information transfer rate of BCI
       It doesn't include eye shift time

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.
    Tw : float
        Signal duration (in second).

    Returns
    -------
    itr: float
        Theoretical ITR (bits/min).
    """

    if y_true.size != y_pred.size:
        raise ValueError(
            """The size of the predicted label and the real label should be the same""")
    # Calculate the number of commands
    M = np.unique(y_true).size
    P = metrics.accuracy_score(y_true, y_pred)
    if P == 1:
        P = P - 0.0001  # Avoid special cases
    # Calculate ITR
    itr = np.log2(M) + P * np.log2(P) + (1 - P) * np.log2(((1 - P) / (M - 1)))
    itr = itr * 60 / Tw

    return itr


def _practical_itr(y_true: ndarray, y_pred: ndarray, Tw: float, Ts: float) -> float:
    """Practical information transfer rate of BCI
       It includes eye shift time

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.
    Tw : float
        Signal duration (in second).
    Ts : float
        Eye shift time (in second).

    Returns
    -------
    itr: float
        Theoretical ITR (bits/min).
    """

    if y_true.size != y_pred.size:
        raise ValueError(
            """The size of the predicted label and the real label should be the same""")
    # Calculate the number of commands
    M = np.unique(y_true).size
    P = metrics.accuracy_score(y_true, y_pred)
    if P == 1:
        P = P - 0.0001  # Avoid special cases
    # Calculate ITR
    itr = np.log2(M) + P * np.log2(P) + (1 - P) * np.log2(((1 - P) / (M - 1)))
    itr = itr * 60 / (Tw + Ts)

    return itr


def _confusion_matrix(y_true: ndarray, y_pred: ndarray, isdraw=False) -> ndarray:
    """Compute confusion matrix to evaluate the accuracy

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.
    isdraw: bool
        draw picture of confusion matrix

    Returns
    -------
    matrix: ndarray
        confusion_matrix (n_class, n_class).
    """

    if y_true.size != y_pred.size:
        raise ValueError(
            """The size of the predicted label and the real label should be the same""")

    matrix = metrics.confusion_matrix(y_true, y_pred)

    if isdraw:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot()
        plt.show()

    return matrix


def _indicators(y_true: ndarray, y_pred: ndarray) -> typing.Tuple[ndarray, Any, Any, Any]:
    """Compute indicators(TP, FP, FN, TN) of confusion matrix

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    TP, FP, FN, TN: ndarray(n_class,)
        indicators of confusion matrix for all classes.
    """

    matrix = _confusion_matrix(y_true, y_pred)
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    return TP, FP, FN, TN


def _tpr_count(y_true: ndarray, y_pred: ndarray) -> int:
    """Sensitivity, hit rate, recall, or true positive rate(TPR)

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    tpr: int
        true positive rate(TPR).
    """

    TP, FP, FN, TN = _indicators(y_true, y_pred)
    tpr = TP / (TP + FN)
    # Average all classes
    return tpr.mean()


def _fnr_count(y_true: ndarray, y_pred: ndarray) -> int:
    """False negative rate(FNR)

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    fnr: int
        False negative rate(FNR).
    """

    TP, FP, FN, TN = _indicators(y_true, y_pred)
    fnr = FN / (TP + FN)
    # Average all classes
    return fnr.mean()


def _fpr_count(y_true: ndarray, y_pred: ndarray) -> int:
    """Fall out or false positive rate (FPR)

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    fpr: int
        false positive rate (FPR).
    """

    TP, FP, FN, TN = _indicators(y_true, y_pred)
    fpr = FP / (FP + TN)
    # Average all classes
    return fpr.mean()


def _tnr_count(y_true: ndarray, y_pred: ndarray) -> int:
    """Specificity or true negative rate (TNR)

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels.

    Returns
    -------
    tnr: int
        true negative rate (TNR).
    """

    TP, FP, FN, TN = _indicators(y_true, y_pred)
    tnr = TN / (TN + FP)
    # Average all classes
    return tnr.mean()


def _roc_auc(y_true: ndarray, y_score: ndarray, isdraw=False) -> ndarray:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_score:  array-like of (n_samples, n_classes)
        Target scores.
    isdraw: bool
        draw picture of ROC

    Returns
    -------
    auc: float
        Area Under the Curve score.
    """

    # Converting decision coefficients to probabilities [1]
    # [1] https://doi.org/10.1088/1741-2552/ab914e
    y_score = np.exp(y_score)
    sum_sample = y_score.sum(axis=1, keepdims=True)
    sum_sample = np.tile(sum_sample, [1, y_score.shape[1]])
    y_score_new = y_score / sum_sample

    # AUC
    auc = metrics.roc_auc_score(y_true, y_score_new, average='macro', multi_class='ovr')

    # ROC
    if isdraw:
        if np.size(np.unique(y_true)) != 2:
            raise ValueError(
                """Only the binary classification task can plot ROC curves""")
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score_new)
        plt.plot(fpr, tpr)
        plt.show()

    return auc


estimators = {
    "Acc": _accuracy,
    "bAcc": _balance_accuracy,
    "tITR": _theoretical_itr,
    "pITR": _practical_itr,
    "TPR": _tpr_count,
    "FNR": _fnr_count,
    "FPR": _fpr_count,
    "TNR": _tnr_count,
    "AUC": _roc_auc,
}


def _check_est(est):
    """Check if a given estimator is valid.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    est : callable object or str
        Could be the name of estimator or a callable estimator itself.

    Returns
    -------
    est: callable object
        A callable estimator.
    """
    if callable(est):
        pass
    elif est in estimators.keys():
        est = estimators[est]
    else:
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function"""
            % (est, (" , ").join(estimators.keys()))
        )
    return est


class Performance(BaseEstimator, TransformerMixin):
    """Evaluation of BCI performance.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    Tw : float
        Signal duration (in second).
    Ts : float
        Eye shift time (in second).
    estimators_list : list
        **supported estimators**
            `Acc`: Accuracy classification score.\n
            `bAcc`: balanced accuracy to deal with imbalanced datasets.\n
            `tITR`: theoretical ITR.\n
            `pITR`: practical ITR.\n
            `TPR`: true positive rate(TPR).\n
            `FNR`: false negative rate(FNR).\n
            `FPR`: false positive rate (FPR).\n
            `TNR`: true negative rate (TNR).\n
            `AUC`: Area under the curve.\n
    isdraw : bool
        Whether to draw the ROC curve.

    Attributes
    ----------
    estimators_list : list
        **supported estimators**
            `Acc`: Accuracy classification score.\n
            `bAcc`: balanced accuracy to deal with imbalanced datasets.\n
            `tITR`: theoretical ITR.\n
            `pITR`: practical ITR.\n
            `TPR`: true positive rate(TPR).\n
            `FNR`: false negative rate(FNR).\n
            `FPR`: false positive rate (FPR).\n
            `TNR`: true negative rate (TNR).\n
            `AUC`: Area under the curve.\n
    Tw : float
        Signal duration (in second).
    Ts : float
        Eye shift time (in second).
    isdraw : bool
        Whether to draw the ROC curve.

    Tip
    ----------
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: Example

        1.from metabci.brainda.utils.performance import Performance.\n
        2.performance = Performance(estimators_list=["Acc","pITR","TPR","AUC"], Tw=0.5, Ts=0.5).\n
        3.results = performance.evaluate(y_true=y[test_ind], y_pred=p_labels, y_score=p_corr).\n
    """

    def __init__(self, estimators_list=["Acc", "pITR"], Tw=None, Ts=None, isdraw=False):
        self.estimators_list = estimators_list
        self.Tw = Tw
        self.Ts = Ts
        self.isdraw = isdraw
        # Check if the parameters are enough
        if "tITR" in self.estimators_list:
            if Tw is None:
                raise ValueError(
                    """theoretical ITR requires Signal duration(Tw)""")
        if "pITR" in self.estimators_list:
            if Tw is None or Ts is None:
                raise ValueError(
                    """practical ITR requires Signal duration(Tw) and Eye shift time(Tw) """)

    def evaluate(self, y_true, y_pred, y_score=None):
        """Transform EEG to covariance matrix.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        y_true : 1d array-like
            Ground truth (correct) labels.
        y_pred : 1d array-like
            Predicted labels.
        y_score:  array-like of (n_samples, n_classes)
            Target scores.

        Returns
        -------
        results: list
            Evaluate the results and form a dictionary.
        """
        results = dict()
        # Iterate through all estimator
        for estimator in self.estimators_list:

            # check y_score
            if estimator == "AUC" and y_score is None:
                raise ValueError(
                    """AUC requires target scores (y_score)""")
            # check estimator
            est = _check_est(estimator)

            # count metrics
            if estimator in ["Acc", "bAcc", "TPR", "FNR", " FPR", "TNR"]:
                res = est(y_true, y_pred)
            elif estimator == "tITR":
                res = est(y_true, y_pred, Tw=self.Tw)
            elif estimator == "pITR":
                res = est(y_true, y_pred, Tw=self.Tw, Ts=self.Ts)
            elif estimator == "AUC":
                res = est(y_true, y_score, isdraw=self.isdraw)

            results[estimator] = res

        return results
