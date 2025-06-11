# -*- coding: utf-8 -*-

"""We apply SEntropy, FEntropy, DEntropy and PSEntropy, after data
pre-processing in the version, to reflect the time series complexity in the
EEG signal.

This module implements four entropy-based methods for quantifying
the complexity of EEG time series signals:

1. Sample Entropy (SEntropy) - Measures regularity by
   quantifying the probability that similar patterns remain similar
   when the dimension increases.
2. Fuzzy Entropy (FEntropy) - An improved version of sample entropy using fuzzy
   membership functions for more robust measurements [2]_.
3. Distribution Entropy (DEntropy) - Estimates complexity by analyzing the
   probability distribution of distances between embedded vectors [3]_.
4. Power Spectral Entropy (PSEntropy) - Computes entropy from the
   power spectraldensity of the signal [4]_.

Typical Usage:

1. Initialize entropy estimator (e.g., SEntropy())
2. Fit to EEG data (n_trials × n_channels × n_samples)
3. Transform to get entropy features
4. Visualize results using draw() method

References:

.. [1] http://en.wikipedia.org/wiki/Sample_Entropy.
.. [2] Cao Z, Lin C T. Inherent fuzzy entropy for the
    improvement of EEG complexity evaluation[J]. IEEE Transactions
    on Fuzzy Systems, 2017, 26(2): 1032-1035.
.. [3] Li P, Liu C, Li K, et al.Assessing the complexity of
    short-term heartbeat interval series by distribution entropy[J].
    Medical & Biological Engineering & Computing, 2015, 53(1): 77-87.
.. [4] A. Zhang, B. Yang and L. Huang, "Feature Extraction of EEG Signals
    Using Power Spectral Entropy," 2008 International Conference on BioMedical
    Engineering and Informatics, Sanya, China, 2008, pp. 435-439

"""

import numpy as np
from numpy import ndarray
from scipy import spatial
from sklearn.base import BaseEstimator, TransformerMixin
import mne
import matplotlib.pyplot as plt


def topoplot(feature, sfreq, chan_names, chan_types='eeg', vmax=None,
             vmin=None, headsize=0.05, verbose=None, axes=None, show=False,
             cmap='RdBu_r'):
    """ Brain topography.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-2-6

    update log:
        2023-2-6 by Baolian shan <baolianshan@tju.edu.cn>

    Parameters
    ----------
    feature: 1d-array, shape(n_channel)
        The input feature.
    sfreq: float
        The sampling rate.
    chan_names: list
        The name of channels.
    chan_types: str
        The type of channel, by default 'eeg'
    headsize: float
        The head size(radius, in meters) to use for spherical montages.
    verbose: bool, str, or int
        Control verbosity of the logging output. If None, use the default
        verbosity level.
    vmax: float
        The value specifying the upper bound of the color range.
    vmin: float
        The value specifying the lower bound of the color range.
    show: bool
        Show the figure if True.
    axes: axes
        The axes to plot to, by default 'None'. If None, a new Figure
        will be created.
    cmap: matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.

    Returns
    -------
    im: matplotlib.image.AxesImage
        The interpolated data.

    References
    ----------
    .. [1] https://mne.tools/0.15/generated/mne.viz.plot_topomap.html.

    """
    # Kwarg** = head_size
    montage_type = mne.channels.make_standard_montage(
        'standard_1020', head_size=headsize)
    # Kwarg** = chan_names, chan_types, sfreq, verbose
    epoch_info = mne.create_info(
        ch_names=chan_names, ch_types=chan_types,
        sfreq=sfreq, verbose=verbose)
    epoch_info.set_montage(montage_type)
    # Kwarg** = feature, axes, show, vmax, vmin
    im, cn = mne.viz.plot_topomap(
        data=feature, pos=epoch_info, axes=axes,
        show=show, vlim=(vmin, vmax), cmap=cmap)
    return im


def _embed(data, order, delay):
    """ Time-delay embedding.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-2-6

    update log:
        2023-2-6 by Baolian shan <baolianshan@tju.edu.cn>

    Parameters
    ----------
    data: 1d-array, shape(n_times)
        Time series.
    order: int
        Embedding dimension(order).
    delay: int
        Delay.

    Returns
    -------
    embedded: ndarray-like, shape(order, n_times - (order - 1) * delay)
        Embedded time-series.

    """
    N = len(data)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = data[i * delay:i * delay + Y.shape[1]]
    return Y


def _tempmatch(data, entropy='SEntropy', metric='chebyshev',
               tolerance=None, gradient=None):
    """ Calculate the degree of template matching between
    matrix column vectors.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-2-6

    update log:
        2023-2-6 by Baolian shan <baolianshan@tju.edu.cn>

    Parameters
    ----------
    data: ndarray-like, shape(order, n_times - (order - 1) * delay)
        Data features after time-delayed embedding.
    entropy: str or function, optional.
        Categories of entropy, 'SEntropy' and 'SEntropy',
        by default 'SEntropy'.
    metric: str or function, optional
        The distance metric to use. The distance function can be "chebyshev",
        "euclidean"......, by default 'chebyshev'.
    tolerance: float
        Similarity tolerance.
    gradient: int
        Boundary gradient(entropy='FEntropy').

    Returns
    -------
    tempmatched: float
        Matched feature matrix.

    """

    A = spatial.distance.pdist(data.T, metric)
    # The total number of two pairs of column vectors
    # l=(1+2+3+...+(n_times-(order-1)*delay))/2
    length = len(A)
    if entropy == 'SEntropy':
        tempmatched = np.sum(A < tolerance) / length
    elif entropy == 'FEntropy':
        tempmatched = np.sum(
            np.exp(-np.power(A, gradient) / tolerance)) / length
    else:
        raise ValueError(
            """ % s is not an valid method ! Valid methods are: SEntropy,
            FEntropy or a callable function"""
            % (entropy)
        )
    return tempmatched


class SEntropy(BaseEstimator, TransformerMixin):
    """ Sample Entropy(SEntropy).

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-2-6

    update log:
        2023-2-6 by Baolian shan <baolianshan@tju.edu.cn>

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs to run (default=None).
    deaverage : bool, optional
        Whether to deaverage the data (default=False).
    order : int, optional
        Embedding dimension (default=2).
    delay : int, optional
        Time delay for embedding (default=1).
    tolerance : ndarray-like, optional
        Similarity tolerance with shape (n_channels, n_times, 1)
        (default=None).
    figsize : tuple, optional
        Figure dimensions as (width, height) in inches (default=(8, 4)).
    sfreq : int, optional
        Sampling frequency in Hz (default=None).
    chan_names : list, optional
        List of channel names (default=None).
    headsize : float, optional
        Head size radius in meters for spherical montages (default=0.05).
    cmap: str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize_title : int, optional
        Font size for title (default=20).
    fontweight_title : str, optional
        Font weight for title ('normal' or 'bold') (default='bold').
    fontcolor_title : str, optional
        Font color for title (default='black').
    fontstyle_title : str, optional
        Font style for title (default='normal').
    loc_title : str, optional
        Title location (default='center').
    pad_title : int, optional
        Title padding in points (default=15).
    fontsize_colorbar : int, optional
        Font size for colorbar label (default=15).
    fontweight_colorbar : str, optional
        Font weight for colorbar label (default='bold').
    fontcolor_colorbar : str, optional
        Font color for colorbar label (default='black').
    fontstyle_colorbar : str, optional
        Font style for colorbar label (default='normal').
    label_colorbar : str, optional
        Label text for colorbar (default='SEntropy').
    loc_colorbar : str, optional
        Location for colorbar label (default='center').

    Raises
    ------
    ValueError
        None

    Note
    ----
        SEntropy generally does not require deaveraging.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Sample_Entropy.

    """

    def __init__(self, n_jobs=None, deaverage=False,
                 order=2, delay=1, tolerance=None,
                 figsize=(8, 4), sfreq=None, chan_names=None,
                 headsize=0.05, cmap: object = 'RdBu_r',
                 fontsize_title=20, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='normal',
                 loc_title='center', pad_title=15,
                 fontsize_colorbar=15, fontweight_colorbar='bold',
                 fontcolor_colorbar='black',
                 fontstyle_colorbar='normal',
                 label_colorbar='SEntropy', loc_colorbar='center'):
        self.n_jobs = n_jobs
        self.deaverage = deaverage
        self.order = order
        self.delay = delay
        self.tolerance = tolerance

        # The variables of analysis:
        self.figsize = figsize
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.headsize = headsize
        self.cmap = cmap

        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.loc_title = loc_title
        self.pad_title = pad_title

        self.fontsize_colorbar = fontsize_colorbar
        self.fontweight_colorbar = fontweight_colorbar
        self.fontcolor_colorbar = fontcolor_colorbar
        self.fontstyle_colorbar = fontstyle_colorbar
        self.label_colorbar = label_colorbar
        self.loc_colorbar = loc_colorbar

    def fit(self, X: ndarray, y: ndarray):
        """ Initialize model parameters by analyzing EEG data
        structure and class labels.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples)
        y : ndarray
            Class labels for each trial with shape (n_trials,).

        """

        self.n_channels = X.shape[1]
        self.classes_ = np.unique(y)
        self.index = [y == label for label in self.classes_]
        return self

    def transform(self, X: ndarray):
        """Compute SEntropy for EEG trials using phase-space embedding.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples)

        Returns
        -------
        ndarray
            entropy values with shape (n_trials, n_channels) where:
            - Higher values indicate more irregular/less predictable signals
            - Lower values suggest more periodic/structured activity

        """

        # SEntropy generally does not require deaveraging.
        if self.deaverage is True:
            X = (X - np.mean(X, axis=-1, keepdims=True)) / \
                np.std(X, axis=-1, keepdims=True)
        if self.tolerance is None:
            tolerance = 0.1 * np.std(X, axis=-1)

        n_trials = X.shape[0]
        X_entropy = np.empty((n_trials, self.n_channels))
        for trail in range(n_trials):
            for channel in range(self.n_channels):
                embed_1 = _embed(X[trail, channel, :],
                                 order=self.order, delay=self.delay)
                tempmatched_1 = _tempmatch(embed_1, entropy='SEntropy',
                                           metric='chebyshev',
                                           tolerance=tolerance[trail, channel])
                # Increase the embedding dimension by 1 unit.
                embed_2 = _embed(X[trail, channel, :],
                                 order=self.order + 1, delay=self.delay)
                tempmatched_2 = _tempmatch(embed_2, entropy='SEntropy',
                                           metric='chebyshev',
                                           tolerance=tolerance[trail, channel])
                entropy = - np.log(tempmatched_2 / tempmatched_1)
                X_entropy[trail, channel] = entropy
        self.X_entropy = X_entropy
        return self.X_entropy

    def draw(self):
        """Visualize SEntropy topography for two EEG conditions.

        Returns
        -------
        ndarray
            Combined entropy values with shape (2, n_channels) containing:
            - Row 0: Mean entropy for class_0 trials
            - Row 1: Mean entropy for class_1 trials

        """

        X_entropy_epoch0 = self.X_entropy[self.index[0]]
        X_entropy_epoch1 = self.X_entropy[self.index[1]]
        X_entropy_combine = np.concatenate([
            np.mean(X_entropy_epoch0, axis=0, keepdims=True),
            np.mean(X_entropy_epoch1, axis=0, keepdims=True)
        ], axis=0)
        vmax = np.max(X_entropy_combine)
        vmin = np.min(X_entropy_combine)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        i = 0
        for axes_col, class_name in zip(axes, ('class_0', 'class_1')):
            im = topoplot(X_entropy_combine[i, :], sfreq=self.sfreq,
                          chan_names=self.chan_names, chan_types='eeg',
                          vmax=vmax, vmin=vmin, headsize=self.headsize,
                          verbose=None, axes=axes_col, show=False,
                          cmap=self.cmap)
            # The format of the title can be designed to suit your needs.
            # References:
            # https://blog.csdn.net/weixin_42297390/article/details/113035057
            axes_col.set_title(class_name, loc=self.loc_title,
                               pad=self.pad_title,
                               fontsize=self.fontsize_title,
                               fontweight=self.fontweight_title,
                               color=self.fontcolor_title,
                               fontstyle=self.fontstyle_title)
            i = +1
        cb = fig.colorbar(im, ax=[axes[0], axes[1]])
        cb.set_label(label=self.label_colorbar, loc=self.loc_colorbar,
                     fontsize=self.fontsize_colorbar,
                     fontweight=self.fontweight_colorbar,
                     color=self.fontcolor_colorbar,
                     fontstyle=self.fontstyle_colorbar)
        plt.show()
        return X_entropy_combine


class FEntropy(BaseEstimator, TransformerMixin):
    """ Fuzzy Entropy(FEntropy).

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-2-6

    update log:
        2023-2-6 by Baolian shan <baolianshan@tju.edu.cn>

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs to run (default=None).
    deaverage : bool, optional
        Whether to deaverage the data (default=False).
    order : int, optional
        Embedding dimension (default=2).
    delay : int, optional
        Time delay for embedding (default=1).
    tolerance : ndarray-like, optional
        Similarity tolerance with shape (n_channels, n_times, 1)
        (default=None).
    gradient : int, optional
        Order of gradient to apply (default=2).
    figsize : tuple, optional
        Figure dimensions as (width, height) in inches (default=(8, 4)).
    sfreq : int, optional
        Sampling frequency in Hz (default=None).
    chan_names : list, optional
        List of channel names (default=None).
    headsize : float, optional
        Head size radius in meters for spherical montages (default=0.05).
    cmap: str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize_title : int, optional
        Font size for title (default=20).
    fontweight_title : str, optional
        Font weight for title ('normal' or 'bold') (default='bold').
    fontcolor_title : str, optional
        Font color for title (default='black').
    fontstyle_title : str, optional
        Font style for title (default='normal').
    loc_title : str, optional
        Title location (default='center').
    pad_title : int, optional
        Title padding in points (default=15).
    fontsize_colorbar : int, optional
        Font size for colorbar label (default=15).
    fontweight_colorbar : str, optional
        Font weight for colorbar label (default='bold').
    fontcolor_colorbar : str, optional
        Font color for colorbar label (default='black').
    fontstyle_colorbar : str, optional
        Font style for colorbar label (default='normal').
    label_colorbar : str, optional
        Label text for colorbar (default='FEntropy').
    loc_colorbar : str, optional
        Location for colorbar label (default='center').

    Raises
    ------
    ValueError
        None

    Note
    ----
        FEntropy generally requires deaveraging.

    References
    ----------
    .. [1] Cao Z, Lin C T. Inherent fuzzy entropy for the improvement
        of EEG complexity evaluation[J].
        IEEE Transactions on Fuzzy Systems, 2017, 26(2): 1032-1035.

    """

    def __init__(self, n_jobs=None, deaverage=True, order=2,
                 delay=1, tolerance=None, gradient=2,
                 figsize=(8, 4), sfreq=None, chan_names=None,
                 headsize=0.05, cmap='RdBu_r',
                 fontsize_title=20, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='normal',
                 loc_title='center', pad_title=15,
                 fontsize_colorbar=15, fontweight_colorbar='bold',
                 fontcolor_colorbar='black',
                 fontstyle_colorbar='normal',
                 label_colorbar='FEntropy',
                 loc_colorbar='center'):
        self.n_jobs = n_jobs
        self.deaverage = deaverage
        self.order = order
        self.delay = delay
        self.tolerance = tolerance
        self.gradient = gradient

        # The variables of analysis:
        self.figsize = figsize
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.headsize = headsize
        self.cmap = cmap

        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.loc_title = loc_title
        self.pad_title = pad_title

        self.fontsize_colorbar = fontsize_colorbar
        self.fontweight_colorbar = fontweight_colorbar
        self.fontcolor_colorbar = fontcolor_colorbar
        self.fontstyle_colorbar = fontstyle_colorbar
        self.label_colorbar = label_colorbar
        self.loc_colorbar = loc_colorbar

    def fit(self, X: ndarray, y: ndarray):
        """ Initialize model parameters by analyzing EEG data
        structure and class labels.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):
        y : ndarray
            Class labels for each trial with shape (n_trials,).

        """

        # X.shape: (n_trials, n_channels, n_samples)
        self.n_channels = X.shape[1]
        self.classes_ = np.unique(y)
        self.index = [y == label for label in self.classes_]
        return self

    def transform(self, X: ndarray):
        """Compute FEntropy for EEG trials using phase-space embedding.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):

        Returns
        -------
        ndarray
            entropy values with shape (n_trials, n_channels) where:
            - Higher values indicate more irregular/less predictable signals
            - Lower values suggest more periodic/structured activity

        """

        # FEntropy generally requires deaveraging!
        if self.deaverage is True:
            X = (X - np.mean(X, axis=-1, keepdims=True)) / \
                np.std(X, axis=-1, keepdims=True)
        if self.tolerance is None:
            tolerance = 0.1 * np.std(X, axis=-1)

        n_trials = X.shape[0]
        X_entropy = np.empty((n_trials, self.n_channels))
        for trail in range(n_trials):
            for channel in range(self.n_channels):
                embed_1 = _embed(X[trail, channel, :],
                                 order=self.order, delay=self.delay)
                embed_1 = embed_1 - np.mean(embed_1, axis=0, keepdims=True)
                tempmatched_1 = _tempmatch(embed_1, entropy='FEntropy',
                                           metric='chebyshev',
                                           tolerance=tolerance[trail, channel],
                                           gradient=self.gradient)
                # Increase the embedding dimension by 1 unit
                embed_2 = _embed(X[trail, channel, :],
                                 order=self.order + 1, delay=self.delay)
                embed_2 = embed_2 - np.mean(embed_2, axis=0, keepdims=True)
                tempmatched_2 = _tempmatch(embed_2, entropy='FEntropy',
                                           metric='chebyshev',
                                           tolerance=tolerance[trail, channel],
                                           gradient=self.gradient)
                entropy = - np.log(tempmatched_2 / tempmatched_1)
                X_entropy[trail, channel] = entropy
        self.X_entropy = X_entropy
        return self.X_entropy

    def draw(self):
        """Visualize FEntropy topography for two EEG conditions.

        Returns
        -------
        ndarray
            Combined entropy values with shape (2, n_channels) containing:
            - Row 0: Mean entropy for class_0 trials
            - Row 1: Mean entropy for class_1 trials

        """

        X_entropy_epoch0 = self.X_entropy[self.index[0]]
        X_entropy_epoch1 = self.X_entropy[self.index[1]]
        X_entropy_combine = np.concatenate([
            np.mean(X_entropy_epoch0, axis=0, keepdims=True),
            np.mean(X_entropy_epoch1, axis=0, keepdims=True)
        ], axis=0)
        vmax = np.max(X_entropy_combine)
        vmin = np.min(X_entropy_combine)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        i = 0
        for axes_col, class_name in zip(axes, ('class_0', 'class_1')):
            im = topoplot(X_entropy_combine[i, :], sfreq=self.sfreq,
                          chan_names=self.chan_names, chan_types='eeg',
                          vmax=vmax, vmin=vmin, headsize=self.headsize,
                          verbose=None, axes=axes_col, show=False,
                          cmap=self.cmap)
            # The format of the title can be designed to suit your needs
            axes_col.set_title(class_name, loc=self.loc_title,
                               pad=self.pad_title,
                               fontsize=self.fontsize_title,
                               fontweight=self.fontweight_title,
                               color=self.fontcolor_title,
                               fontstyle=self.fontstyle_title)
            i = +1
        cb = fig.colorbar(im, ax=[axes[0], axes[1]])
        cb.set_label(label=self.label_colorbar, loc=self.loc_colorbar,
                     fontsize=self.fontsize_colorbar,
                     fontweight=self.fontweight_colorbar,
                     color=self.fontcolor_colorbar,
                     fontstyle=self.fontstyle_colorbar)
        plt.show()
        return X_entropy_combine


class DEntropy(BaseEstimator, TransformerMixin):
    """ Distribution Entropy(DEntropy).

    author: chen guowei <chen_guowei2000@tju.edu.cn>

    Created on: 2023-4-10

    update log:
        2023-4-10 by chen guowei <chen_guowei2000@tju.edu.cn>

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs to run (default=None).
    deaverage : bool, optional
        Whether to deaverage the data (default=False).
    order : int, optional
        Embedding dimension (default=2).
    delay : int, optional
        Time delay for embedding (default=1).
    bin_num : int, optional
        Number of bins for histogram-based entropy (default=10).
    metric : str, optional
        Distance metric for entropy calculation
        ('chebyshev', 'euclidean', etc.) (default='chebyshev').
    eps : float, optional
        Small constant to avoid division by zero (default=1e-10).
    figsize : tuple, optional
        Figure dimensions as (width, height) in inches (default=(8, 4)).
    sfreq : int, optional
        Sampling frequency in Hz (default=None).
    chan_names : list, optional
        List of channel names (default=None).
    headsize : float, optional
        Head size radius in meters for spherical montages (default=0.05).
    cmap: str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize_title : int, optional
        Font size for title (default=20).
    fontweight_title : str, optional
        Font weight for title ('normal' or 'bold') (default='bold').
    fontcolor_title : str, optional
        Font color for title (default='black').
    fontstyle_title : str, optional
        Font style for title (default='normal').
    loc_title : str, optional
        Title location (default='center').
    pad_title : int, optional
        Title padding in points (default=15).
    fontsize_colorbar : int, optional
        Font size for colorbar label (default=15).
    fontweight_colorbar : str, optional
        Font weight for colorbar label (default='bold').
    fontcolor_colorbar : str, optional
        Font color for colorbar label (default='black').
    fontstyle_colorbar : str, optional
        Font style for colorbar label (default='normal').
    label_colorbar : str, optional
        Label text for colorbar (default='DEntropy').
    loc_colorbar : str, optional
        Location for colorbar label (default='center').

    Raises
    ------
    ValueError
        None

    References
    ----------
    .. [1] Li P, Liu C, Li K, et al.Assessing the complexity of
        short-term heartbeat interval series by distribution entropy[J].
        Medical & Biological Engineering & Computing, 2015, 53(1): 77-87.

    """

    def __init__(self, n_jobs=None, deaverage=True, order=2, delay=1,
                 bin_num=10, metric='chebyshev', eps=1e-10,
                 figsize=(8, 4), sfreq=None, chan_names=None,
                 headsize=0.05, cmap='RdBu_r',
                 fontsize_title=20, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='normal',
                 loc_title='center', pad_title=15,
                 fontsize_colorbar=15, fontweight_colorbar='bold',
                 fontcolor_colorbar='black',
                 fontstyle_colorbar='normal',
                 label_colorbar='DEntropy',
                 loc_colorbar='center'):

        self.n_jobs = n_jobs
        self.deaverage = deaverage
        self.order = order
        self.delay = delay
        self.bin_num = bin_num
        self.metric = metric
        self.eps = eps

        self.figsize = figsize
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.headsize = headsize
        self.cmap = cmap

        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.loc_title = loc_title
        self.pad_title = pad_title

        self.fontsize_colorbar = fontsize_colorbar
        self.fontweight_colorbar = fontweight_colorbar
        self.fontcolor_colorbar = fontcolor_colorbar
        self.fontstyle_colorbar = fontstyle_colorbar
        self.label_colorbar = label_colorbar
        self.loc_colorbar = loc_colorbar

    def fit(self, X: ndarray, y: ndarray):
        """ Initialize model parameters by analyzing EEG data
        structure and class labels.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):
        y : ndarray
            Class labels for each trial with shape (n_trials,).

        """

        self.n_channels = X.shape[1]
        self.classes_ = np.unique(y)
        self.index = [y == label for label in self.classes_]
        return self

    def transform(self, X: ndarray):
        """Compute DEntropy for EEG trials using phase-space embedding.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):

        Returns
        -------
        ndarray
            entropy values with shape (n_trials, n_channels) where:
            - Higher values indicate more irregular/less predictable signals
            - Lower values suggest more periodic/structured activity

        """

        if self.deaverage:
            X = (X - np.mean(X, axis=-1, keepdims=True)) / \
                np.std(X, axis=-1, keepdims=True)

        n_trials = X.shape[0]
        X_entropy = np.empty((n_trials, self.n_channels))

        for trial in range(n_trials):
            for channel in range(self.n_channels):

                # Time-delay embedding
                embed = _embed(X[trial, channel, :],
                               order=self.order, delay=self.delay)

                # Calculate distance
                dist = spatial.distance.pdist(embed.T, self.metric)

                # Compute histogram probability
                hist, _ = np.histogram(dist, bins=self.bin_num, density=True)
                prob = hist / hist.sum()

                # Calculate Shannon entropy (add eps to avoid log(0) errors)
                entropy = -np.sum(prob * np.log(prob + self.eps))
                entropy /= np.log(self.bin_num)
                X_entropy[trial, channel] = entropy

        self.X_entropy = X_entropy
        return self.X_entropy

    def draw(self):
        """Visualize DEntropy topography for two EEG conditions.

        Returns
        -------
        ndarray
            Combined entropy values with shape (2, n_channels) containing:
            - Row 0: Mean entropy for class_0 trials
            - Row 1: Mean entropy for class_1 trials

        """

        X_entropy_epoch0 = self.X_entropy[self.index[0]]
        X_entropy_epoch1 = self.X_entropy[self.index[1]]
        X_entropy_combine = np.concatenate([
            np.mean(X_entropy_epoch0, axis=0, keepdims=True),
            np.mean(X_entropy_epoch1, axis=0, keepdims=True)
        ], axis=0)

        vmax = np.max(X_entropy_combine)
        vmin = np.min(X_entropy_combine)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        i = 0
        for axes_col, class_name in zip(axes, ('class_0', 'class_1')):
            im = topoplot(X_entropy_combine[i, :], sfreq=self.sfreq,
                          chan_names=self.chan_names, chan_types='eeg',
                          vmax=vmax, vmin=vmin, headsize=self.headsize,
                          verbose=None, axes=axes_col, show=False,
                          cmap=self.cmap)
            # The format of the title can be designed to suit your needs
            axes_col.set_title(class_name, loc=self.loc_title,
                               pad=self.pad_title,
                               fontsize=self.fontsize_title,
                               fontweight=self.fontweight_title,
                               color=self.fontcolor_title,
                               fontstyle=self.fontstyle_title)
            i = +1
        cb = fig.colorbar(im, ax=[axes[0], axes[1]])
        cb.set_label(label=self.label_colorbar, loc=self.loc_colorbar,
                     fontsize=self.fontsize_colorbar,
                     fontweight=self.fontweight_colorbar,
                     color=self.fontcolor_colorbar,
                     fontstyle=self.fontstyle_colorbar)
        plt.show()
        return X_entropy_combine


class PSEntropy(BaseEstimator, TransformerMixin):
    """Power Spectral Entropy (PSEntropy).

    author: chen guowei <chen_guowei2000@tju.edu.cn>

    Created on: 2023-4-10

    update log:
        2023-4-10 by chen guowei <chen_guowei2000@tju.edu.cn>

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs to run (default=None).
    deaverage : bool, optional
        Whether to deaverage the data (default=False).
    n_fft : int, optional
        Length of FFT window
        (default=None, auto-determined from signal length).
    eps : float, optional
        Small constant to avoid division by zero (default=1e-10).
    bin_num : int, optional
        Number of bins for histogram-based entropy (default=10).
    metric : str, optional
        Distance metric for entropy calculation
        ('chebyshev', 'euclidean', etc.) (default='chebyshev').
    eps : float, optional
        Small constant to avoid division by zero (default=1e-10).
    figsize : tuple, optional
        Figure dimensions as (width, height) in inches (default=(8, 4)).
    sfreq : int, optional
        Sampling frequency in Hz (default=None).
    chan_names : list, optional
        List of channel names (default=None).
    headsize : float, optional
        Head size radius in meters for spherical montages (default=0.05).
    cmap: str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize_title : int, optional
        Font size for title (default=20).
    fontweight_title : str, optional
        Font weight for title ('normal' or 'bold') (default='bold').
    fontcolor_title : str, optional
        Font color for title (default='black').
    fontstyle_title : str, optional
        Font style for title (default='normal').
    loc_title : str, optional
        Title location (default='center').
    pad_title : int, optional
        Title padding in points (default=15).
    fontsize_colorbar : int, optional
        Font size for colorbar label (default=15).
    fontweight_colorbar : str, optional
        Font weight for colorbar label (default='bold').
    fontcolor_colorbar : str, optional
        Font color for colorbar label (default='black').
    fontstyle_colorbar : str, optional
        Font style for colorbar label (default='normal').
    label_colorbar : str, optional
        Label text for colorbar (default='PSEntropy').
    loc_colorbar : str, optional
        Location for colorbar label (default='center').

    Raises
    ------
    ValueError
        None

    References
    ----------
    .. [1] A. Zhang, B. Yang and L. Huang, "Feature Extraction of
        EEG Signals Using Power Spectral Entropy,".
        2008 International Conference on BioMedical Engineering
        and Informatics, Sanya, China, 2008, pp. 435-439

    """

    def __init__(self, n_jobs=None, deaverage=False,
                 n_fft=None, eps=1e-10,
                 figsize=(8, 4), sfreq=None,
                 chan_names=None, headsize=0.05, cmap='RdBu_r',
                 fontsize_title=20, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='normal',
                 loc_title='center', pad_title=15,
                 fontsize_colorbar=15, fontweight_colorbar='bold',
                 fontcolor_colorbar='black',
                 fontstyle_colorbar='normal',
                 label_colorbar='PSEntropy',
                 loc_colorbar='center'):
        self.deaverage = deaverage
        self.n_fft = n_fft
        self.sfreq = sfreq
        self.eps = eps

        self.figsize = figsize
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.headsize = headsize
        self.cmap = cmap

        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.loc_title = loc_title
        self.pad_title = pad_title

        self.fontsize_colorbar = fontsize_colorbar
        self.fontweight_colorbar = fontweight_colorbar
        self.fontcolor_colorbar = fontcolor_colorbar
        self.fontstyle_colorbar = fontstyle_colorbar
        self.label_colorbar = label_colorbar
        self.loc_colorbar = loc_colorbar

    def fit(self, X: ndarray, y: ndarray):
        """ Initialize model parameters by analyzing
        EEG data structure and class labels.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):
        y : ndarray
            Class labels for each trial with shape (n_trials,).

        """

        self.n_channels = X.shape[1]
        self.classes_ = np.unique(y)
        self.index = [y == label for label in self.classes_]
        return self

    def transform(self, X: ndarray):
        """Compute PSEntropy for EEG trials using phase-space embedding.

        Parameters
        ----------
        X : ndarray
            Input EEG data array with shape (n_trials, n_channels, n_samples):

        Returns
        -------
        ndarray
            entropy values with shape (n_trials, n_channels) where:
            - Higher values indicate more irregular/less predictable signals
            - Lower values suggest more periodic/structured activity

        """

        if self.deaverage is True:
            X = (X - np.mean(X, axis=-1, keepdims=True)) / \
                np.std(X, axis=-1, keepdims=True)

        n_trials, n_channels = X.shape[0], X.shape[1]
        X_entropy = np.empty((n_trials, n_channels))
        for trial in range(n_trials):
            for channel in range(n_channels):

                # Compute FFT
                data_0 = X[trial, channel, :]
                data_fft = np.fft.fft(data_0, n=self.n_fft)

                # Calculate Power Spectral Density (PSD)
                N = len(data_0)
                PSD = np.abs(data_fft) ** 2 / N

                # Normalize PSD to get probability distribution
                PSD_total = np.sum(PSD)
                prob = PSD / PSD_total

                # Compute Shannon entropy (with epsilon to prevent log(0))
                entropy = -np.sum(prob * np.log(prob + self.eps))
                X_entropy[trial, channel] = entropy

        self.X_entropy = X_entropy
        return X_entropy

    def draw(self):
        """Visualize PSEntropy topography for two EEG conditions.

        Returns
        -------
        ndarray
            Combined entropy values with shape (2, n_channels) containing:
            - Row 0: Mean entropy for class_0 trials
            - Row 1: Mean entropy for class_1 trials

        """

        X_entropy_epoch0 = self.X_entropy[self.index[0]]
        X_entropy_epoch1 = self.X_entropy[self.index[1]]
        X_entropy_combine = np.concatenate([
            np.mean(X_entropy_epoch0, axis=0, keepdims=True),
            np.mean(X_entropy_epoch1, axis=0, keepdims=True)], axis=0)
        vmax = np.max(X_entropy_combine)
        vmin = np.min(X_entropy_combine)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        i = 0
        for axes_col, class_name in zip(axes, ('class_0', 'class_1')):
            im = topoplot(X_entropy_combine[i, :], sfreq=self.sfreq,
                          chan_names=self.chan_names, chan_types='eeg',
                          vmax=vmax, vmin=vmin, headsize=self.headsize,
                          verbose=None, axes=axes_col, show=False,
                          cmap=self.cmap)
            axes_col.set_title(class_name, loc=self.loc_title,
                               pad=self.pad_title,
                               fontsize=self.fontsize_title,
                               fontweight=self.fontweight_title,
                               color=self.fontcolor_title,
                               fontstyle=self.fontstyle_title)
            i = +1
        cb = fig.colorbar(im, ax=[axes[0], axes[1]])
        cb.set_label(label=self.label_colorbar, loc=self.loc_colorbar,
                     fontsize=self.fontsize_colorbar,
                     fontweight=self.fontweight_colorbar,
                     color=self.fontcolor_colorbar,
                     fontstyle=self.fontstyle_colorbar)
        plt.show()
        return X_entropy_combine
