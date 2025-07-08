"""
 Deep4Net.
 Modified from https://github.com/braindecode/braindecode/blob/master/braindecode/models/deep4.py

"""


import numpy as np
from torch import nn, Tensor
from torch.nn import init
from torch.nn.functional import elu

from .base import (
    Expression,
    AvgPool2dWithConv,
    Ensure4d,
    identity,
    transpose_time_to_spat,
    squeeze_final_output,
    np_to_th,
    SkorchNet,
)


@SkorchNet  # TODO: Bug Fix required:  unable to make docs with this wrapper
class Deep4Net(nn.Sequential):
    """
    DeepNet was inspired by the successful neural network architecture in computer vision.
    DeepNet has two convolutional layers similar to ShallowNet to handle temporal convolution and
    spatial filtering. [1]_
    In addition to these two convolutional layers,
    DeepNet has improved its learning capability by adding three additional convolutional layers
    and a maximum pooling layer.
    DeepNet also leverages the batch normalization and dropout layers to accelerate and avoid over-fitting during model
    training. DeepNet employs an exponential linear unit (ELU) as the activation function.

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
    >>> estimator = Deep4Net(X.shape[1], X.shape[2], num_classes)
    >>> estimator.fit(X[train_index], y[train_index])

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730


    """

    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes,
    ):
        super().__init__()
        final_conv_length = "auto"
        n_filters_time = 25
        n_filters_spat = 25
        filter_time_length = 10
        pool_time_length = 3
        pool_time_stride = 3
        n_filters_2 = 50
        filter_length_2 = 10
        n_filters_3 = 100
        filter_length_3 = 10
        n_filters_4 = 200
        filter_length_4 = 10
        first_nonlin = elu
        first_pool_mode = "max"
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = "max"
        later_pool_nonlin = identity
        drop_prob = 0.5
        # double_time_convs = False
        split_first_layer = True
        batch_norm = True
        batch_norm_alpha = 0.1
        stride_before_pool = False

        if final_conv_length == "auto":
            assert n_samples is not None
        """self.in_chans = n_channels
        self.n_classes = n_classes
        self.input_window_samples = n_samples
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool"""

        if stride_before_pool:
            conv_stride = pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = pool_time_stride
        self.add_module("ensuredims", Ensure4d())

        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[first_pool_mode]
        later_pool_class = pool_class_dict[later_pool_mode]
        if split_first_layer:
            self.add_module("dimshuffle", Expression(transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    n_filters_time,
                    (filter_time_length, 1),
                    stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    n_filters_time,
                    n_filters_spat,
                    (1, n_channels),
                    stride=(conv_stride, 1),
                    bias=not batch_norm,
                ),
            )
            n_filters_conv = n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    n_channels,
                    n_filters_time,
                    (filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not batch_norm,
                ),
            )
            n_filters_conv = n_filters_time
        if batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        self.add_module("conv_nonlin", Expression(first_nonlin))
        self.add_module(
            "pool",
            first_pool_class(
                kernel_size=(pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        self.add_module("pool_nonlin", Expression(first_pool_nonlin))

        def add_conv_pool_block(n_filters_before, n_filters, filter_length, block_nr):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=drop_prob))
            self.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not batch_norm,
                ),
            )
            if batch_norm:
                self.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            self.add_module("nonlin" + suffix, Expression(later_nonlin))

            self.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            self.add_module("pool_nonlin" + suffix, Expression(later_pool_nonlin))

        add_conv_pool_block(n_filters_conv, n_filters_2, filter_length_2, 2)
        add_conv_pool_block(n_filters_2, n_filters_3, filter_length_3, 3)
        add_conv_pool_block(n_filters_3, n_filters_4, filter_length_4, 4)

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if final_conv_length == "auto":
            out = self(
                np_to_th(
                    np.ones(
                        (1, n_channels, n_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_4,
                n_classes,
                (final_conv_length, 1),
                bias=True,
            ),
        )
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()

    def cal_backbone(self, X: Tensor, **kwargs):
        tmp = X
        for i in range(len(self)-1):
            tmp = self[i](tmp)
        return tmp
