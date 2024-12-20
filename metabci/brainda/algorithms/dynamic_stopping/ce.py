# -*- coding: utf-8 -*-
"""
This module contains the implementation of the CE (Cross-Entropy) class.

Classes:
    CE: A class for handling dynamic stopping algorithms using cross-entropy based criteria.

Authors: Duan Shunguo<dsg@tju.edu.cn>

Date: 2024/9/1

"""

import numpy as np
from sklearn.base import clone, BaseEstimator, TransformerMixin
import joblib


class CE(BaseEstimator, TransformerMixin):
    """
    A class for handling dynamic stopping algorithms using cross-entropy based criteria.

    Attributes:
        decoder: The decoder for EEG to be used.
        model_dict (dict): A dictionary to store models.
        n_classes (int): Number of classification classes.
        user_mode (int): Mode of the user, 0 for normal, 1 for saving model text file.

    Methods:
        _save_model(filename): Saves the model to a file.
        _load_model(filename): Loads the model from a file.
        _cross_entropy(rho_i): Computes the cross-entropy cost.
        _get_model(duration): Retrieves the model information for a given duration.
        fit(X, Y, duration, Yf=None, filename=None): Trains the model using the provided data.
        predict(data, duration, t_max=1, thre=-1.5*1e-3, filename=None): Makes a decision based on the provided data and model.

    Example:
        >>> ce = CE(decoder, n_classes=3)
        >>> ce.fit(X, Y, duration)
        >>> decision, label = ce.predict(data, duration)
    """

    def __init__(self, decoder, n_classes, user_mode=0):
        """
        Initializes the CE class with the given decoder, number of classes, and user mode.

        Parameters:
            decoder: The decoder for EEG to be used.
            n_classes (int): Number of classes.
            user_mode (int): Mode of the user, 0 for normal, 1 for saving model text file.
        """
        self.decoder = decoder
        self.model_dict = {}
        self.n_classes = n_classes
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

    def _cross_entropy(self, rho_i):
        """
        Computes the cross-entropy cost.

        Parameters:
            rho_i (dict): Input coefficient of correlation data.

        Returns:
            tuple: Cost for hypothesis H0 and cost for hypothesis Hq.
        """
        n = self.n_classes
        rho_q = np.array([[np.partition(rho_i[i], -1)[-1],
                           np.partition(rho_i[i], -2)[-2]] for i in rho_i])
        cost_h0 = np.sum(rho_i[0]) - n * np.log(np.sum(np.exp(rho_i[0])))
        cost_hq = np.array([rho_q[i, 0] - rho_q[i, 1]
                           for i, _ in enumerate(rho_q)])
        return cost_h0, cost_hq

    def _get_model(self, duration):
        """
        Retrieves the model information for a given duration.

        Parameters:
            duration (float): Duration for which the model is trained.

        Returns:
            estimator: The estimator for the given duration.
        """
        model_info = self.model_dict[duration]
        estimator = model_info['estimator']
        return estimator

    def fit(self, X, Y, duration, Yf=None, filename=None):
        """
        Trains the model using the provided data.

        Parameters:
            X (array-like): Training data.
            Y (array-like): Training labels.
            duration (float): Duration for which the model is trained.
            Yf (array-like, optional): Additional training data. Defaults to None.
            filename (str, optional): File name to save the model. Defaults to None.

        Returns:
            estimator: The trained estimator.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        data = X
        label = Y
        yf = Yf
        estimator = clone(self.decoder).fit(data, label, Yf=yf)
        self.model_dict[duration] = {"estimator": estimator}

        if self.user_mode == 1 and filename is not None:
            self._save_model(filename)
        return estimator

    def predict(self, data, duration, t_max=1, thre=-1.5 * 1e-3, filename=None):
        """
        Makes a decision based on the provided data and model.

        Parameters:
            data (array-like): Input data.
            duration (float): Duration for which the model is used.
            t_max (float): Maximum duration for the model.
            thre (float, optional): Threshold for decision making. Defaults to -1.5*1e-3.
            filename (str, optional): File name to load the model. Defaults to None.

        Returns:
            tuple: Decision (True/False) and predicted label.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        elif self.user_mode == 1 and filename is not None:
            self._load_model(filename)

        if duration in self.model_dict:
            estimator = self._get_model(duration)
            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i, _ in enumerate(rhos)}
            cost_h0, cost_hq = self._cross_entropy(rho_i)

            if -cost_h0 * thre > -cost_hq or duration >= t_max:
                return True, label
            else:
                return False, label
        else:
            raise ValueError(f"No model found for duration: {duration}")
