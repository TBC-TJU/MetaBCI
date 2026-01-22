# -*- coding: utf-8 -*-
"""
This module implements four brain network feature
analysis algorithms for quantifying functional
connectivity or directed information flow in EEG signals:

1. Phase Locking Value (PLV) - Measures phase synchronization
between signals by estimating the stability of phase differences across trials.
2. Phase Lag Index (PLI) - Quantifies asymmetric phase relationship
distribution to reduce volume conduction effects in connectivity analysis.
3. Weighted Phase Lag Index (WPLI) - Improves PLI robustness by
introducing phase-difference magnitude weighting to suppress noise influence.
4. Partial Directed Coherence (PDC) - Analyzes directed causal
influences between multivariate time series in the frequency domain.

References
--------
.. [1] Wang, G. et al. Epileptic Seizure Detection Based
    on Partial Directed Coherence Analysis. IEEE Journal of Biomedical
    and Health Informatics 20, 873–879 (2016).
"""

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
import mne
from joblib import Parallel, delayed


def matrixplot(feature, xticklabels, yticklabels, axes, cmap='Blues',
               vmax=None, vmin=None, fontsize=10, fontweight='bold',
               fontcolor='black', fontstyle='normal'):
    """ Basic matrix plot drawing function.

    author: Baolian shan <baolianshan@tju.edu.cn>

    updata log:
        2025-3-28 by Ruinan Zhou <rainy_zhou@tju.edu.cn>

    Parameters
    ----------
    feature : ndarray, shape (n_channels, n_channels)
        The input feature.
    xticklabels : list
        The list of string labels of x-axis.
    yticklabels : list
        The list of string labels of y-axis.
    axes : axes
        The axes to plot to, by default 'None'. If None,
        a new Figure will be created.
    cmap : matplotlib colormap
        Colormap to use. If None, defaults to 'Blue'.
    vmin : float
        The value specifying the lower bound of the color range.
    vmax : float
        The value specifying the upper bound of the color range.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are 'blue',
        'purple', 'orange', 'green', 'red', 'yellow', 'pink', 'lightgreen',
        'forestgreen', 'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.

    Returns
    -------
    mp : matplotlib object
        The matrix plot object.
    """
    mp = axes.matshow(
        feature,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        interpolation='nearest')
    axes.set_xticks(np.arange(0, len(xticklabels)))
    axes.set_xticklabels(xticklabels, fontsize=fontsize,
                         fontweight=fontweight, color=fontcolor,
                         fontstyle=fontstyle)
    axes.set_yticks(np.arange(0, len(yticklabels)))
    axes.set_yticklabels(yticklabels, fontsize=fontsize,
                         fontweight=fontweight, color=fontcolor,
                         fontstyle=fontstyle)
    return mp


def Topomapplot(feature, axes, channelnames, sfreq, cmap='Blues',
                channeltypes='eeg', montage='standard_1020'):
    """ Basic topmap plot drawing function.

    author:Ruinan Zhou<Ruinan Zhou <rainy_zhou@tju.edu.cn>

    created on 2025-3-28

    Parameters
    ----------
    feature : ndarray, shape (n_segments, n_channels)
        The input feature.
    axes : axes
        The axes to plot to, by default 'None'.
        If None, a new Figure will be created.
    channelnames : list
        The name of channels.
    sfreq : int
        The sampling frequency.
    cmap : matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are 'Reds',
        'Pink', 'Blues', 'Purples', 'Oranges', 'Greys', 'Greens', 'GnBu',
        'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn', 'YlGnBu',
        'coolwarm_r', 'coolwarm'.
    channeltypes : str
        The type of channels, by default 'eeg'.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.

    Returns
    -------
    tp : matplotlib object
        The topomap plot object.
    """
    info = mne.create_info(
        ch_names=channelnames,
        sfreq=sfreq,
        ch_types=channeltypes)
    montage = mne.channels.make_standard_montage(montage)
    info.set_montage(montage)
    evoked = mne.EvokedArray(feature.reshape(-1, 1), info, tmin=0)
    tp, cn = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        cmap=cmap,
        axes=axes,
        show=False,
        contours=False
    )
    return tp


class BaseBNA(BaseEstimator, TransformerMixin):
    """ Base class for brain network analysis.

    author:Ruinan Zhou <rainy_zhou@tju.edu.cn>

    created on 2025-3-28

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (reserved parameter).
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.
    cmap : str, matplotlib colormap
        Colormap to use. Defaults to 'Blue'.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are 'blue',
        'purple', 'orange', 'green', 'red', 'yellow', 'pink', 'lightgreen',
        'forestgreen', 'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    loc_title : str
        The location of title, supported values are 'center', 'left', 'right'.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location, supported values are
        'center', 'left', 'right'.


    Attributes
    ----------
    n_jobs : int
        Number of parallel jobs.
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data.
    cmap : str, matplotlib colormap
        Colormap to use..
    fontsize : float or str
        The font size.
    fontcolor : str
        The font color.
    fontstyle : str
        The font style.
    loc_title : str
        The location of title.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location.

    Note
    ----
    The class is designed to be inherited by other classes,
    and should not be used directly.

    """

    def __init__(self, n_jobs=1, sfreq=1000, figsize=(16, 6),
                 chan_names=None, montage='standard_1020', cmap='Blues',
                 fontsize_ticks=12, fontweight_ticks='bold',
                 fontcolor_ticks='black', fontstyle_ticks='normal',
                 fontsize_title=20, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='normal',
                 loc_title='center', pad_title=15, fontsize_colorbar=15,
                 fontweight_colorbar='bold', fontcolor_colorbar='black',
                 fontstyle_colorbar='normal', label_colorbar='Sync Value',
                 loc_colorbar='center'):

        self.n_jobs = n_jobs
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.figsize = figsize
        self.cmap = cmap
        self.montage = montage
        self.fontsize_ticks = fontsize_ticks
        self.fontweight_ticks = fontweight_ticks
        self.fontcolor_ticks = fontcolor_ticks
        self.fontstyle_ticks = fontstyle_ticks
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
        """Get class labels and channel numbers.

        author: Baolian shan<baolianshan@tju.edu.cn>

        Parameters
        ---------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            The input data.
        y : ndarray, shape (n_trials,)
            The class labels of the input data.

        Returns
        -------
        self : object
            Returns self.
        """
        # X.shape: (n_trials, n_channels, n_samples)
        self.n_channels = X.shape[1]
        self.classes_ = np.unique(y)
        self.index = [y == label for label in self.classes_]
        return self

    def _draw(self, epochs, plot_type='matrix', nrows=1, ncols=2):
        """ Basic drawing function for brain network analysis.

        Parameters
        ----------
        epochs : ndarray, shape (n_classes, n_trials,
        feature_size_X, feature_size_Y)
            All kinds of characteristic data to be drawn
        plot_type : str
            The type of plot, supported values are 'matrix' and 'topomap'.
        nrows : int, optional
            Number of subgraph rows (default: 1 row)
        ncols : int, optional
            Number of subgraph columns (if None, it is automatically
            calculated as ceil(number of categories /nrows))

        Returns
        -------
        X_combine : ndarray,shape (n_classes, feature_size_X, feature_size_Y)
            Combined class average feature.
        """
        if self.chan_names is None:
            raise ValueError("Channel names are not provided.")
        # Combine the epochs of each class
        X_combine = np.concatenate([np.mean(
            epochs[self.index[i]], axis=0, keepdims=True)
            for i in range(len(self.classes_))], axis=0)
        n_classes = X_combine.shape[0]
        # Dynamically calculate the number of rows and columns for subplots
        ncols = ncols if ncols else int(np.ceil(n_classes / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(
            self.figsize[0] * ncols, self.figsize[1] * nrows))
        axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
        # Uniform color range
        vmax = np.max(X_combine)
        vmin = np.min(X_combine)
        # Draw the matrix for each class
        for i, ax in enumerate(axes):
            if i >= n_classes:
                ax.axis('off')
                print(f"Class {i} does not exist, skipping...")
                continue
            if plot_type == 'matrix':
                # Draw the matrix
                plot = matrixplot(
                    X_combine[i],
                    xticklabels=self.chan_names,
                    yticklabels=self.chan_names,
                    axes=ax, cmap=self.cmap,
                    vmax=vmax, vmin=vmin,
                    fontsize=self.fontsize_ticks,
                    fontweight=self.fontweight_ticks,
                    fontcolor=self.fontcolor_ticks,
                    fontstyle=self.fontstyle_ticks
                )
            elif plot_type == 'topomap':
                # Draw the topomap
                plot = Topomapplot(
                    X_combine[i],
                    axes=ax,
                    channelnames=self.chan_names,
                    sfreq=self.sfreq,
                    cmap=self.cmap,
                    channeltypes='eeg',
                    montage=self.montage
                )
            ax.set_title(
                "Class " + str(self.classes_[i]),
                loc=self.loc_title, pad=self.pad_title,
                fontsize=self.fontsize_title, fontweight=self.fontweight_title,
                color=self.fontcolor_title, fontstyle=self.fontstyle_title
            )
        # Set the colorbar and its label
        cb = fig.colorbar(plot, ax=axes.tolist())
        cb.set_label(
            label=self.label_colorbar,
            loc=self.loc_colorbar,
            fontsize=self.fontsize_colorbar,
            fontweight=self.fontweight_colorbar,
            color=self.fontcolor_colorbar,
            fontstyle=self.fontstyle_colorbar
        )
        return X_combine


class PLV(BaseBNA):
    """ Phase Locking Value (PLV)

    We apply PLV after data pre-processing in the version,
    to reflect the average coherence of EEG signals.

    author: Baolian shan <baolianshan@tju.edu.cn>

    updata log:
        2025-3-28 by Ruinan Zhou <rainy_zhou@tju.edu.cn>

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (reserved parameter).
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.
    cmap : str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are 'Reds', 'Pink',
        'Blues', 'Purples', 'Oranges', 'Greys', 'Greens',
        'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu','coolwarm_r', 'coolwarm'.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are 'blue',
        'purple', 'orange', 'green', 'red', 'yellow', 'pink', 'lightgreen',
        'forestgreen', 'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    loc_title : str
        The location of title, supported values are 'center', 'left', 'right'.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location, supported values are
        'center', 'left', 'right'.

    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using PLV

        from metabci.brainda.algorithms.feature_analysis.
        brain_network_analysisimport PLV
        plv = PLV(n_jobs=None, figsize=(16, 6),
                chan_names=chan_names, cmap ='Blues',
                fontsize_ticks=12, fontweight_ticks='bold',
                fontcolor_ticks='black', fontstyle_ticks='normal',
                fontsize_title=20, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='normal',
                loc_title='center', pad_title=15,
                fontsize_colorbar=20, fontweight_colorbar='bold',
                fontcolor_colorbar='black', fontstyle_colorbar='normal',
                label_colorbar='PLV', loc_colorbar='center')
        plv.fit(X, y)
        X_plv = plv.transform(X)
        plv.draw()

    """

    def __init__(self, **kwargs):
        kwargs.setdefault('label_colorbar', 'PLV')
        super().__init__(**kwargs)

    def transform(self, X: ndarray):
        """ Extract PLV features from eeg data

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            The input data.

        Returns
        -------
        X_plv : ndarray, shape (n_trials, n_channels, n_channels)
            The PLV matrix for each trials.
        """
        n_trails = X.shape[0]
        self.X_plv_matrix = np.ones(
            (n_trails, self.n_channels, self.n_channels))
        X_plv = np.ones(
            (n_trails, int(self.n_channels * (self.n_channels - 1) / 2)))

        for trial in range(n_trails):
            value: np.ndarray = np.array([])
            for channel in range(self.n_channels):
                instant_phase_channel = np.angle(
                    signal.hilbert(X[trial, channel, :]))
                for compared_channel in range(channel + 1, self.n_channels):
                    instant_phase_compared_channel = np.angle(
                        signal.hilbert(X[trial, compared_channel, :]))
                    phase_differ = np.abs(
                        instant_phase_channel -
                        instant_phase_compared_channel)
                    re = np.cos(phase_differ)
                    im = np.sin(phase_differ)
                    plv = np.sqrt(np.sum(re)**2 + np.sum(im) **
                                  2) / np.shape(phase_differ)[0]
                    self.X_plv_matrix[trial, channel, compared_channel] = plv
                    self.X_plv_matrix[trial, compared_channel, channel] = plv
                    value = np.append(value, plv)
            X_plv[trial, :] = value
        self.X_plv = X_plv
        return self.X_plv

    def draw(self):
        """ Draw the PLV matrix of the epochs"""
        self._draw(self.X_plv_matrix, plot_type='matrix')


class PLI(BaseBNA):
    """ Phase Lag Index (PLI)

    The PLI is a measure of phase synchronization between two signals,
    which is less sensitive to volume conduction effects than the PLV.

    author: Ruinan Zhou <rainy_zhou@tju.edu.cn>

    created on: 2025-3-28

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (reserved parameter).
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.
    cmap : str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy',
        'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are
        'blue', 'purple', 'orange', 'green', 'red', 'yellow',
        'pink', 'lightgreen', 'forestgreen', 'cyan', 'teal',
        'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    loc_title : str
        The location of title, supported values are 'center', 'left', 'right'.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location, supported values
        are 'center', 'left', 'right'.

    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using PLI

        from metabci.brainda.algorithms.feature_analysis.
        brain_network_analysis import PLI
        pli = PLI(n_jobs=None, figsize=(16, 6),
                chan_names=chan_names, cmap ='Blues',
                fontsize_ticks=12, fontweight_ticks='bold',
                fontcolor_ticks='black', fontstyle_ticks='normal',
                fontsize_title=20, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='normal',
                loc_title='center', pad_title=15,
                fontsize_colorbar=20, fontweight_colorbar='bold',
                fontcolor_colorbar='black', fontstyle_colorbar='normal',
                label_colorbar='PLI', loc_colorbar='center')
        pli.fit(X, y)
        X_pli = pli.transform(X)
        pli.draw()

    """

    def __init__(self, **kwargs):
        kwargs.setdefault('label_colorbar', 'PLI')
        super().__init__(**kwargs)

    def transform(self, X: np.ndarray):
        """Extract features from all segments

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            The input data.

        Returns
        -------
        X_pli : ndarray, shape (n_trials, n_channels*(n_channels-1)/2)
            The PLI matrix for each trials.

        """
        n_trials = X.shape[0]

        self.X_pli_matrix = np.zeros(
            (n_trials, self.n_channels, self.n_channels))
        X_pli = np.zeros(
            (n_trials, int(self.n_channels * (self.n_channels - 1) / 2)))

        for trial in range(n_trials):
            value = []
            for i in range(self.n_channels):
                phase_i = np.angle(signal.hilbert(X[trial, i, :]))
                for j in range(i + 1, self.n_channels):
                    phase_j = np.angle(signal.hilbert(X[trial, j, :]))
                    phase_diff = phase_i - phase_j
                    signed_imag = np.sign(phase_diff)
                    pli_value = np.abs(np.mean(signed_imag))
                    self.X_pli_matrix[trial, i, j] = pli_value
                    self.X_pli_matrix[trial, j, i] = pli_value
                    value.append(pli_value)
            X_pli[trial, :] = np.array(value)
        return X_pli

    def draw(self):
        """ Draw the PLI matrix of the epochs"""
        self._draw(self.X_pli_matrix, plot_type='matrix')


class WPLI(BaseBNA):
    """ Weighted Phase Lag Index (WPLI)

    The WPLI is a measure of phase synchronization between two signals,
    which is less sensitive to volume conduction effects than the PLV.

    author: Ruinan Zhou <rainy_zhou@tju.edu.cn>

    created on: 2025-3-28

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (reserved parameter).
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.
    cmap : str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r',
        'RdYlGn', 'RdYlGn', 'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large',
        'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are
        'ultralight', 'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold', 'demi',
        'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are
        'blue', 'purple', 'orange', 'green', 'red',
        'yellow', 'pink', 'lightgreen', 'forestgreen',
        'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    loc_title : str
        The location of title, supported values are 'center', 'left', 'right'.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location, supported values are
        'center', 'left', 'right'.

    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using WPLI

        from metabci.brainda.algorithms.feature_analysis.
        brain_network_analysis import WPLI
        wpli = WPLI(n_jobs=None, figsize=(16, 6),
                chan_names=chan_names, cmap ='Blues',
                fontsize_ticks=12, fontweight_ticks='bold',
                fontcolor_ticks='black', fontstyle_ticks='normal',
                fontsize_title=20, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='normal',
                loc_title='center', pad_title=15,
                fontsize_colorbar=20, fontweight_colorbar='bold',
                fontcolor_colorbar='black', fontstyle_colorbar='normal',
                label_colorbar='WPLI', loc_colorbar='center')
        wpli.fit(X, y)
        X_wpli = wpli.transform(X)
        wpli.draw()

    """

    def __init__(self, **kwargs):
        kwargs.setdefault('label_colorbar', 'WPLI')
        super().__init__(**kwargs)

    def transform(self, X: np.ndarray):
        """ Extract features from all segments

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            The input data.

        Returns
        -------
        X_wpli : ndarray, shape (n_trials, n_channels*(n_channels-1)/2)
            The WPLI matrix for each trials.
        """
        n_trials = X.shape[0]
        self.n_channels = X.shape[1]

        self.X_wpli_matrix = np.zeros(
            (n_trials, self.n_channels, self.n_channels))
        X_wpli = np.zeros(
            (n_trials, int(self.n_channels * (self.n_channels - 1) / 2)))

        for trial in range(n_trials):
            value = []
            for i in range(self.n_channels):
                phase_i = np.angle(signal.hilbert(X[trial, i, :]))
                for j in range(i + 1, self.n_channels):
                    phase_j = np.angle(signal.hilbert(X[trial, j, :]))
                    phase_diff = phase_i - phase_j
                    imag_part = np.sin(phase_diff)
                    numerator = np.abs(imag_part)
                    denominator = imag_part
                    epsilon = 1e-10
                    wpli_value = np.abs(
                        np.mean(numerator / (denominator + epsilon)))
                    self.X_wpli_matrix[trial, i, j] = wpli_value
                    self.X_wpli_matrix[trial, j, i] = wpli_value
                    value.append(wpli_value)
            X_wpli[trial, :] = np.array(value)
        return X_wpli

    def draw(self):
        """ Draw the WPLI matrix of the epochs"""
        self._draw(self.X_wpli_matrix, plot_type='matrix')


class PDC(BaseBNA):
    """ Partial Directed Coherence (PDC)

    The PDC is a measure of directed connectivity between two signals,
    which is based on the frequency domain representation of the VAR model.

    author: Ruinan Zhou <rainy_zhou@tju.edu.cn>

    created on: 2025-3-28

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (reserved parameter).
    sfreq : int
        Sampling frequency of the EEG data.
    figsize : tuple
        Dimensions of the length and width of the brain topography.
    chan_names : list
        The name of channels.
    montage : str
        The montage of the EEG data, by default 'standard_1020'.
    cmap : str, matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are
        'Reds', 'Pink', 'Blues', 'Purples', 'Oranges', 'Greys',
        'Greens', 'GnBu', 'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn',
        'YlGnBu', 'coolwarm_r', 'coolwarm'.
    fontsize : float or str
        If the font size is str, supported values are 'xx-small',
        'x-small', 'small', 'medium', 'large',
        'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold',
        'demi', 'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are
        'blue', 'purple', 'orange', 'green', 'red',
        'yellow', 'pink', 'lightgreen', 'forestgreen',
        'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    loc_title : str
        The location of title, supported values are 'center', 'left', 'right'.
    pad_title : float
        The offset of the title from the top of the Axes, in points.
    label_colorbar : str
        The label of colorbar.
    loc_colorbar : str
        The colorbar label location, supported values are
        'center', 'left', 'right'.
    max_order : int
        The maximum order of the VAR model, by default 20.

    Attributes
    ----------
    max_order : int
        The maximum order of the VAR model.
    freq_res_points : int
        The number of frequency resolution points.

    Note
    ----
    Parallel computing is highly recommended; otherwise,
    the computation speed could be significantly slow.

    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using PDC

        from metabci.brainda.algorithms.feature_analysis.
        brain_network_analysis import PDC
        pdc = PDC(max_order=20, n_jobs=None, figsize=(16, 6),
                chan_names=chan_names, cmap ='Blues',
                fontsize_ticks=12, fontweight_ticks='bold',
                fontcolor_ticks='black', fontstyle_ticks='normal',
                fontsize_title=20, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='normal',
                loc_title='center', pad_title=15,
                fontsize_colorbar=20, fontweight_colorbar='bold',
                fontcolor_colorbar='black', fontstyle_colorbar='normal',
                label_colorbar='PDC', loc_colorbar='center')
        pdc.fit(X, y)
        X_pdc = pdc.transform(X)
        pdc.draw()

    """

    def __init__(self, max_order=20, **kwargs):
        kwargs.setdefault('label_colorbar', 'PDC')
        super().__init__(**kwargs)
        self.max_order = max_order
        self.freq_band = (0, self.sfreq / 2)

    def _compute_pdc_matrix(self, segment):
        """ Calculate the PDC matrix for a single segment

        Parameters
        ----------
        segment : ndarray, shape (n_channels, n_samples)
            The input segment data.

        Returns
        -------
        pdc_matrix_f : ndarray, shape (n_freqs, n_channels, n_channels)
            The PDC matrix in the frequency domain.
        """
        data = segment.T
        self.freq_res_points = int(data.shape[0] / 2)
        # Fit VAR model to the data
        model = VAR(data)

        def compute_aic(p):
            return model.fit(p).aic
        aic = Parallel(n_jobs=-1)(
            delayed(compute_aic)(p) for p in range(1, self.max_order + 1))
        optimal_p = np.argmin(aic) + 1
        var_result = model.fit(optimal_p)
        coeffs = var_result.coefs
        # Compute the A(f) matrix
        freqs = np.linspace(
            self.freq_band[0],
            self.freq_band[1],
            self.freq_res_points)
        A_f = np.zeros((len(freqs), self.n_channels, self.n_channels),
                       dtype=np.complex_)

        def compute_A_f(freq):
            Af = np.eye(self.n_channels, dtype=np.complex_)
            for r in range(1, optimal_p + 1):
                Af -= coeffs[r - 1] * np.exp(-2j * np.pi * freq * r)
            return Af
        A_f = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_A_f)(freq) for freq in freqs)
        A_f = np.array(A_f)
        # Compute the PDC matrix
        pdc_matrix_f = np.zeros((len(freqs), self.n_channels, self.n_channels))
        for j in range(self.n_channels):
            denom = np.sqrt(np.sum(np.abs(A_f[:, :, j])**2, axis=1))
            for i in range(self.n_channels):
                numer = abs(A_f[:, i, j])
                pdc_matrix_f[:, i, j] = numer / denom
        return pdc_matrix_f

    def transform(self, X: ndarray):
        """ Extract features from all segments

        Parameters
        ----------
        X : ndarray, shape (n_segments, n_channels, n_samples)
            The input data.

        returns
        -------
        features : ndarray, shape (n_segments, n_channels)
            The flow out information of each channel.
        """
        self.segments = X
        n_segments = X.shape[0]
        self.X_pdc_matrix = np.zeros(
            (n_segments, self.n_channels, self.n_channels))
        self.X_pdc_f_matrix = np.zeros(
            (n_segments, self.n_channels, self.n_channels))
        self.flow_out_features = np.zeros((n_segments, self.n_channels))

        def process_segment(segment):
            pdc_f = self._compute_pdc_matrix(segment)
            pdc = np.sum(pdc_f**2, axis=0)
            return {
                "X_pdc": pdc,
                "flow_out": np.sum(pdc, axis=1)
            }
            pass
        # Parallel processing for each segment
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(process_segment)(self.segments[seg])
            for seg in range(n_segments)
        )
        for segment in range(n_segments):
            self.X_pdc_matrix[segment] = result[segment]['X_pdc']
            self.flow_out_features[segment] = result[segment]['flow_out']
        return self.flow_out_features

    def draw(self):
        """ Draw the PDC matrix and flow out information
            topomap of the epochs
        """
        self._draw(self.X_pdc_matrix, plot_type="matrix")
        self._draw(self.flow_out_features, plot_type="topomap")
