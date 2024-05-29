"""
For source code implemented in pytorch, please refer to https://github.com/Akasuyi/PreTrainingMI
"""

import copy

import torch
from numpy import ndarray
from torch import nn

from .base import (
    SkorchNet,
)


@SkorchNet
class FineTuneNet(nn.Module):
    def __init__(self, backbone, test_classes, size_after_backbone):
        super().__init__()
        self.backbone = backbone
        self.test_head = torch.nn.Linear(size_after_backbone, test_classes)

    def forward(self, x):
        backbone_result = self.backbone.cal_backbone(x)
        fine_tune_result = self.test_head(backbone_result)
        return fine_tune_result


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model


class PreTraing():
    """CrossDataset transfer learning method [1].
    This class implement a simple
    for original code in [1], please refer to http:github

    author: Xie

    Created on: 2024-02-18

    update log:

    Parameters
    ----------
    target_n_class: int
        number of classes in the target dataset.
    size_before_classification: int
        size_before_classification is the feature size
        of the last feature extraction layer(layer before the linear layer)'s output,
        it depends on the inputs length and model structure.

    Attributes
    ----------
    target_n_class: int
        number of classes in the target dataset.
    size_before_classification: int
        size_before_classification is the feature size
        of the last feature extraction layer(layer before the linear layer)'s output,
        it depends on the inputs length and model structure.

    Raises
    ----------

    References
    ----------
    .. [1] Xie, Yuting, et al. "Cross-dataset transfer learning for motor imagery signal classification via
           multi-task learning and pre-training." Journal of Neural Engineering 20.5 (2023): 056037.

    """

    def __init__(self, target_n_class, size_before_classification):
        """
        Initializing some basic parameter
        """
        self.target_n_class = target_n_class
        self.size_before_classification = size_before_classification

    def pretraining(self, model: nn.Module, save_path: str, X: ndarray, y: ndarray):
        """pre-training model using source dataset.

         Parameters
         ----------
         model: nn.Module:
            model to be pre-train
         save_path: str:
            path to save the pre-trained model weight
         X: ndarray
            EEG data in source dataset, shape(n_trials, n_channels, n_samples).
         y：ndarry
            Label, shape(n_trials,).

         """
        model.fit(X, y)
        save_model(model.module_, save_path)

    def finetuning(self, model: nn.Module, save_path: str, X: ndarray, y: ndarray):
        """ fine-tuning model using target dataset

        Parameters
        ----------
        model: nn.Module:
            pre-trained model instance, pre-trained model weight is not required
        save_path: str:
            path to save the pre-trained model weight
        X: ndarray
            EEG data in target dataset, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,).

        Returns
        -------
        model : wrapped model after fine-tuning

        """
        model = copy.deepcopy(model.module)
        model = load_model(model, save_path)
        fine_tune_model = FineTuneNet(model, self.target_n_class, self.size_before_classification)
        fine_tune_model.fit(X, y)
        return fine_tune_model
