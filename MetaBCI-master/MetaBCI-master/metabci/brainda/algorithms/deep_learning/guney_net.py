# -*- coding: utf-8 -*-
"""
Guney's network proposed in A Deep Neural Network for SSVEP-based Brain-Computer Interfaces.

Modified from https://github.com/osmanberke/Deep-SSVEP-BCI.git
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import (
    compute_same_pad2d,
    _narrow_normal_weight_zero_bias,
    compute_out_size,
    SkorchNet,
)


@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class GuneyNet(nn.Module):
    """

    GuneyNet is a neural network specifically designed for the SSVEP task. [1]_
    The SSVEP paradigm has the characteristic of harmonic correspondence,
    and in addition to the frequency signature of the SSVEP can be observed at the fundamental
    frequency of the stimulus, the corresponding can also be observed at the double frequency up to the sixth frequency.
    The responses of different frequencies have different characteristics,
    and the responses of the same stimulus at lower frequencies usually have larger amplitudes.
    But high-octave responses tend to be less disturbed by other ongoing brain activity,
    and they tend to exhibit relatively high signal-to-noise ratios.

    GuneyNet first synthesizes the results of multiple filter sub-bands through a convolutional network layer,
    then extracts the information on multiple electrodes through spatial convolution similar to the previous network,
    uses two temporal convolution layers to extract temporal information,
    and finally uses a linear layer to classify the extracted features.

    author: Xie YT <xyt_998@tju.edu.cn>

    Created on: 2022-07-02

    update log:
        2023-12-11 by MutexD <wudf@tju.edu.cn>


    Parameters
    ----------

    n_channels: int
        Lead count for the input signal.
    n_samples: int
        Sampling points of the input signal. The value equals sampling rate (Hz) * signal duration (s).
    n_classes: int
        The number of classes of input signals to be classified.

    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> num_classes = 2
    >>> num_sub_bands = 3
    >>> estimator = GuneyNet(X.shape[2], X.shape[3], num_classes, num_sub_bands)
    >>> estimator.fit(X[train_index], y[train_index])

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    References
    ----------
    .. [1] Guney O B , Oblokulov M , Ozkan H . A Deep Neural Network for SSVEP-based
       Brain Computer Interfaces[J]. 2020.
    """

    def __init__(self, n_channels, n_samples, n_classes, n_bands):
        # super(GuneyNet, self).__init__()
        super().__init__()

        n_spatial_filters = 120
        spatial_dropout = 0.1
        time1_kernel = 2
        time1_stride = 2
        n_time1_filters = 120
        time1_dropout = 0.1
        time2_kernel = 10
        n_time2_filters = 120
        time2_dropout = 0.95

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_bands = n_bands

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("band_layer", nn.Conv2d(n_bands, 1, (1, 1), bias=False)),
                    ("spatial_layer", nn.Conv2d(1, n_spatial_filters, (n_channels, 1))),
                    ("spatial_dropout", nn.Dropout(spatial_dropout)),
                    (
                        "time1_layer",
                        nn.Conv2d(
                            n_spatial_filters,
                            n_time1_filters,
                            (1, time1_kernel),
                            stride=(1, time1_stride),
                        ),
                    ),
                    ("time1_dropout", nn.Dropout(time1_dropout)),
                    ("relu", nn.ReLU()),
                    (
                        "same_padding",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (
                                    1,
                                    compute_out_size(
                                        n_samples, time1_kernel, stride=time1_stride
                                    ),
                                ),
                                (1, time2_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "time2_layer",
                        nn.Conv2d(
                            n_time1_filters,
                            n_time2_filters,
                            (1, time2_kernel),
                            stride=(1, 1),
                        ),
                    ),
                    ("time2_dropout", nn.Dropout(time2_dropout)),
                    ("flatten", nn.Flatten()),
                    (
                        "fc_layer",
                        nn.Linear(
                            n_time2_filters
                            * compute_out_size(
                                n_samples, time1_kernel, stride=time1_stride
                            ),
                            n_classes,
                        ),
                    ),
                ]
            )
        )
        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _narrow_normal_weight_zero_bias(self)
        nn.init.ones_(self.model[0].weight)
        # MATLAB uses xavier_uniform_ with varaiance 2/(input+output)
        # perhaps this is a mistake in Help document
        nn.init.xavier_normal_(self.model[-1].weight, gain=1)

    def forward(self, X):
        # X: (n_batch, n_bands, n_channels, n_samples)
        out = self.model(X)
        return out
