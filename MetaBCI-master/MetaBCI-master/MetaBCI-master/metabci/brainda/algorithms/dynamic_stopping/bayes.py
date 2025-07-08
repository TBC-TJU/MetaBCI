# -*- coding: utf-8 -*-

"""
This module contains the implementation of the Bayes-based Dynamic Stopping Algorithm and a dummy KDE class.

Classes:
    DummyKDE: A dummy KDE class that uses a DummyClassifier to come up with insufficient negative samples.

    Bayes: Bayes-based Dynamic Stopping Algorithm can determine in real-time whether the signal quality is
    sufficient based on the current length of EEG data and output the result, achieving higher accuracy and
    signal transmission rate in a shorter time.

Authors: Duan Shunguo<dsg@tju.edu.cn>

Date: 2024/9/1

"""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.dummy import DummyClassifier
from sklearn.base import clone, BaseEstimator, TransformerMixin
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
import joblib


class DummyKDE:
    """
    A dummy KDE class that uses a DummyClassifier to come up with insufficient negative samples.

    Attributes:
        dummy (DummyClassifier): A dummy classifier with a constant strategy.

    Methods:
        __call__(x): Predicts the dummy probability for the input data.

    Example:
        >>> kde = DummyKDE(constant=0)
        >>> kde([1, 2, 3])
        array([0, 0, 0])
    """

    def __init__(self, constant=0):
        """
        Initializes the DummyKDE with a constant value for the DummyClassifier.

        Parameters:
            constant (int): The constant value used by the DummyClassifier.
        """
        self.dummy = DummyClassifier(strategy='constant', constant=constant)
        self.dummy.fit(np.zeros((1)), np.zeros(1))

    def __call__(self, x):
        """
        Predicts the dummy probability for the input data.

        Parameters:
            x (array-like): The input data.

        Returns:
            array: The predicted dummy probabilities.
        """
        X = np.array(x).reshape(-1, 1)
        dummy_prob = self.dummy.predict(X)
        return dummy_prob


class Bayes(BaseEstimator, TransformerMixin):
    """
    The Bayes-based Dynamic Stopping Algorithm for handling Bayesian decoding.

    Attributes:
        decoder: The decoder for EEG to be used.
        model_dict (dict): A dictionary to store Estimator, KDE_models and the Prior possibility.
        user_mode (int): Mode of the user, 0 for normal, 1 for saving model text file.

    Methods:
        _save_model(filename): Saves the model to a file.
        _load_model(filename): Loads the model from a file.
        _extract_dm(pred_labels, Y_test, dm_i): Extracts decision metrics from predicted and true labels.
        fit(X, Y, duration, Yf=None, filename=None): Trains the KDE model and estimator using the provided data.
        _get_model(duration): Retrieves the model information for a given duration.
        predict(data, duration, P_thre=0.95, filename=None): Makes a decision based on the provided data and model.

    Example:
        >>> bayes = Bayes(decoder)
        >>> bayes.fit(X, Y, duration)
        >>> decision, label = bayes.predict(data, duration)
    """

    def __init__(self, decoder, user_mode=0):
        """
        Initializes the Bayes class with the given decoder, maximum duration, and user mode.

        Parameters:
            decoder: The decoder for EEG to be used.
            user_mode (int): Mode of the user, 0 for norm, 1 for saving model text file.
        """
        self.decoder = decoder
        self.model_dict = {}
        self.user_mode = user_mode

    def _save_model(self, filename):
        """
        Saves the model to a file.

        Parameters:
            filename (str): File name.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        joblib.dump(self.model_dict, filename)

    def _load_model(self, filename):
        """
        Loads the model from a file.

        Parameters:
            filename (str): File name.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        self.model_dict = joblib.load(filename)

    def _extract_dm(self, pred_labels, Y_test, dm_i):
        """
        Extracts decision metrics from predicted and true labels.

        Parameters:
            pred_labels (array-like): Predicted labels.
            Y_test (array-like): True labels.
            dm_i: Decision metric data.

        Returns:
            dict: A dictionary with 'correct' and 'incorrect' keys.
        """
        extracted = {'correct': [], 'incorrect': []}
        for i, (pred, true) in enumerate(zip(pred_labels, Y_test)):
            if pred == true:
                extracted['correct'].append(dm_i[i])
            else:
                extracted['incorrect'].append(dm_i[i])
        return extracted

    def fit(self, X, Y, duration, Yf=None, filename=None):
        """
        Trains the KDE model and estimator using the provided data.

        Parameters:
            X (array-like): Training data.
            Y (array-like): Training labels.
            duration (float): Duration for which the model is trained.
            Yf (array-like, optional): Additional training data. Defaults to None.
            filename (str, optional): File name to save the model. Defaults to None.

        Returns:
            tuple: KDE models for correct and incorrect decisions, prior probability, and decision metrics.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        data = X
        label = Y
        Yf = Yf
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}  # 初始化空字典
        prob_list = []
        for train_ind, test_ind in spliter.split(data, y=label):
            X_train, Y_train = np.copy(
                data[train_ind]), np.copy(
                label[train_ind])
            X_test, Y_test = np.copy(data[test_ind]), np.copy(label[test_ind])
            model = clone(self.decoder).fit(X_train, Y_train, Yf=Yf)
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i, _ in enumerate(rhos)}

            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels, Y_test, dm_i)
            sub_prob = len(extracted_dm['correct']) / (len(extracted_dm['correct']) +
                                                       len(extracted_dm['incorrect']))
            prob_list.append(sub_prob)
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']
        dm1 = aggregated_dm['incorrect']

        kde0 = gaussian_kde(dm0)
        if len(dm1) > 2:
            kde1 = gaussian_kde(dm1)
        else:
            kde1 = DummyKDE(0)
        prob = np.mean(prob_list)

        estimator = clone(self.decoder).fit(data, label, Yf=Yf)
        self.model_dict[duration] = {
            'kde0': kde0,
            'kde1': kde1,
            'prob': prob,
            "estimator": estimator}

        if self.user_mode == 1 and filename is not None:
            self._save_model(filename)
        return kde0, kde1, prob, dm0, dm1

    def _get_model(self, duration):
        """
        Retrieves the model information for a given duration.

        Parameters:
            duration (float): Duration for which the model is trained.

        Returns:
            tuple: KDE models for correct and incorrect decisions, prior probability, and estimator.
        """
        model_info = self.model_dict[duration]
        kde0 = model_info['kde0']
        kde1 = model_info['kde1']
        prob = model_info['prob']
        estimator = model_info['estimator']
        return kde0, kde1, prob, estimator

    def predict(self, data, duration, t_max=1, P_thre=0.95, filename=None):
        """
        Makes a decision based on the provided data and model.

        Parameters:
            data (array-like): Input data.
            duration (float): Duration for which the model is used.
            t_max (float): Maximum duration for the model.
            P_thre (float, optional): Probability threshold. Defaults to 0.95.
            filename (str, optional): File name to load the model. Defaults to None.

        Returns:
            tuple: Decision (True/False) and predicted label.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        elif self.user_mode == 1 and filename is not None:
            self._load_model(filename)

        if duration in self.model_dict:
            kde0, kde1, prob, estimator = self._get_model(duration)

            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i, _ in enumerate(rhos)}
            dm_i = np.array([rho_i[i][np.argmax(rho_i[i])] for i in rho_i])
            p_H0 = kde0(dm_i)
            p_H1 = kde1(dm_i)
            p_pre = prob * p_H0 / (prob * p_H0 + (1 - prob) * p_H1)
            p_thre = P_thre

            if p_pre >= p_thre or duration >= t_max:
                return True, label
            else:
                return False, label
        else:
            raise ValueError(f"No model found for duration: {duration}")
