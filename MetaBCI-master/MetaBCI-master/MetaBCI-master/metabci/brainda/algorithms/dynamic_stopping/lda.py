# -*- coding: utf-8 -*-
"""
This module contains the implementation of the LDA (Linear Discriminant Analysis) class.

Classes:
    LDA: A class for handling dynamic stopping algorithms using Linear Discriminant Analysis.

Authors: Duan Shunguo<dsg@tju.edu.cn>

Date: 2024/9/1

"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone, BaseEstimator, TransformerMixin
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
import joblib


class LDA(BaseEstimator, TransformerMixin):
    """
    A class for handling dynamic stopping algorithms using Linear Discriminant Analysis.

    Attributes:
        decoder: The decoder for EEG to be used.
        model_dict (dict): A dictionary to store models.
        user_mode (int): Mode of the user, 0 for normal, 1 for user-defined.

    Methods:
        _save_model(filename): Saves the model to a file.
        _load_model(filename): Loads the model from a file.
        _extract_dm(pred_labels, Y_test, dm_i): Extracts decision metrics from predicted and true labels.
        _get_model(duration): Retrieves the model information for a given duration.
        fit(X, Y, duration, Yf=None, filename=None): Trains the model using the provided data.
        predict(data, duration, t_max=1, filename=None): Makes a decision based on the provided data and model.

    Example:
        >>> lda = LDA(decoder)
        >>> lda.fit(X, Y, duration)
        >>> decision, label = lda.predict(data, duration)
        >>> lda._save_model('model.pkl')
    """

    def __init__(self, decoder, user_mode=0):
        """
        Initializes the LDA class with the given decoder and user mode.

        Parameters:
            decoder: The decoder for EEG to be used.
            user_mode (int): Mode of the user, 0 for normal, 1 for saving model text file.
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
            pred_labels (list): Predicted labels.
            Y_test (list): True labels.
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

    def _get_model(self, duration):
        """
        Retrieves the model information for a given duration.

        Parameters:
            duration (float): Duration for which the model is trained.

        Returns:
            tuple: LDA model and estimator.
        """
        model_info = self.model_dict[duration]
        lda_model = model_info['lda_model']
        estimator = model_info['estimator']
        return lda_model, estimator

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
            model: The trained LDA model.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")

        data = X
        label = Y
        yf = Yf
        spliter = EnhancedLeaveOneGroupOut(return_validate=False)
        aggregated_dm = {'correct': [], 'incorrect': []}
        lda = LinearDiscriminantAnalysis()
        for train_ind, test_ind in spliter.split(data, y=label):
            X_train, Y_train = np.copy(
                data[train_ind]), np.copy(
                label[train_ind])
            X_test, Y_test = np.copy(data[test_ind]), np.copy(label[test_ind])
            model = clone(self.decoder).fit(X_train, Y_train, Yf=yf)
            pred_labels = model.predict(X_test)
            rhos = model.transform(X_test)
            rho_i = {i: rhos[i, :] for i, _ in enumerate(rhos)}

            dm_i = np.array([[1, np.partition(rho_i[i], -2)[-2] /
                            np.partition(rho_i[i], -1)[-1]] for i in rho_i])
            extracted_dm = self._extract_dm(pred_labels, Y_test, dm_i)
            for key in aggregated_dm:
                aggregated_dm[key].extend(extracted_dm[key])
        dm0 = aggregated_dm['correct']
        dm1 = aggregated_dm['incorrect']
        train_L = np.concatenate((dm0, dm1), axis=0)
        labels_L = np.concatenate(
            (np.ones(
                len(dm0)), np.zeros(
                len(dm1))), axis=0)
        model = lda.fit(train_L, labels_L)
        estimator = clone(self.decoder).fit(data, label, Yf=yf)
        self.model_dict[duration] = {
            'lda_model': model, "estimator": estimator}

        if self.user_mode == 1 and filename is not None:
            self._save_model(filename)
        return model

    def predict(self, data, duration, t_max=1, filename=None):
        """
        Makes a decision based on the provided data and model.

        Parameters:
            data (array-like): Input data.
            duration (float): Duration for which the model is used.
            t_max (float): Maximum duration for the model.
            filename (str, optional): File name to load the model. Defaults to None.

        Returns:
            tuple: Decision (True/False) and predicted label.
        """
        if self.user_mode == 1 and filename is None:
            raise ValueError("Filename must be provided when user_mode is 1")
        elif self.user_mode == 1 and filename is not None:
            self._load_model(filename)

        if duration in self.model_dict:
            lda_model, estimator = self._get_model(duration)

            rhos = estimator.transform(data)
            label = estimator.predict(data)
            rho_i = {i: rhos[i, :] for i, _ in enumerate(rhos)}
            dm_i = np.array([[1, np.partition(rho_i[i], -2)[-2] /
                            np.partition(rho_i[i], -1)[-1]] for i in rho_i])

            L = lda_model.predict(dm_i)
            if L == 1 or duration >= t_max:
                return True, label
            else:
                return False, label
        else:
            raise ValueError(f"No model found for duration: {duration}")
