# -*- coding: utf-8 -*-
"""
Time-frequency analysis, including continuous wavelet transform,
short-time Fourier transform, and Hilbert transform.
"""

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
import matplotlib.pyplot as plt


def spectplot(times, freqs, feature, cmap='viridis',
              vmin=None, vmax=None, xlabel='Time(s)',
              ylabel='Frequency(Hz)', text_label='Power (\\muV^2/Hz)',
              fontsize_labels=10, fontweight_labels='bold',
              fontcolor_labels='black', fontstyle_labels='normal',
              fontsize_ticks=10, fontweight_ticks='bold',
              fontcolor_ticks='black', fontstyle_ticks='normal',
              fontsize_title=14, fontweight_title='bold',
              fontcolor_title='limegreen', fontstyle_title='oblique',
              title_label="Spectral Analysis", distance=0.3,
              fontsize_bar=10, fontweight_bar='bold',
              fontcolor_bar='black', fontstyle_bar='normal'):
    """ Spectral analysis.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-02-24

    update log:
        None

    Parameters
    ----------
    times : ndarray
        Array of segment times.
    freqs : ndarray
        Array of sample frequencies.
    feature : 2d-ndarray
        The input feature.
    cmap : matplotlib colormap
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r', supported values are 'Reds',
        'Pink', 'Blues', 'Purples', 'Oranges', 'Greys', 'Greens', 'GnBu',
        'GnBu_r', 'OrRd', 'OrRd_r', 'RdYlGn', 'RdYlGn', 'YlGnBu',
        'coolwarm_r', 'coolwarm'.
    vmin : float
        The value specifying the lower bound of the color range.
    vmax : float
        The value specifying the upper bound of the color range.
    label : str
        The title of the time-frequency graph.
    distance : float
        The horizontal distance between the time-frequency graph and
        the title of colorbar.
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


    References
    ----------
    .. [1] https://blog.csdn.net/m0_51623564/article/details/124747212.
    """

    if vmin is None:
        vmin = np.min(feature)
        vmax = np.max(feature)

    plt.pcolormesh(times, freqs, feature, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels,
               color=fontcolor_labels, fontstyle=fontstyle_labels)
    plt.ylabel(ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels,
               color=fontcolor_labels, fontstyle=fontstyle_labels)
    plt.xticks(
        fontsize=fontsize_ticks,
        fontweight=fontweight_ticks,
        color=fontcolor_ticks,
        fontstyle=fontstyle_ticks)
    plt.yticks(
        fontsize=fontsize_ticks,
        fontweight=fontweight_ticks,
        color=fontcolor_ticks,
        fontstyle=fontstyle_ticks)
    plt.xlim(np.array(0, np.max(times)))
    plt.ylim(np.array(1, np.max(freqs)))
    plt.title(
        title_label,
        fontsize=fontsize_title,
        fontweight=fontweight_title,
        color=fontcolor_title,
        fontstyle=fontstyle_title,
    )
    plt.text(
        np.max(times) + distance,
        np.max(freqs) / 2,
        text_label,
        fontsize=fontsize_bar,
        fontweight=fontweight_bar,
        color=fontcolor_bar,
        fontstyle=fontstyle_bar,
        rotation=90,
        verticalalignment='center',
        horizontalalignment='center',
    )
    plt.colorbar()


def _htplot(
    times, sfreq, signal, envelope, frequency, phase,
    title="Hilbert Transform", fontsize_title=12, fontweight_title='bold',
    fontcolor_title='black', fontstyle_title='oblique',
    signal_label='orignal signal', envelope_label='amplitude envelope',
    phase_label='instantaneous phase',
    frequency_label='instantaneous frequency',
    axis_label='Time(s)',
    signal_color='blue', envelope_color='orange',
    frequency_color='red', phase_color='green',
    fontsize=10, fontweight='bold', fontcolor='black', fontstyle='normal',
    linewidth=1, linestyle='solid',
    bbox_to_anchor=(0.88, 0.33), labelcolor='black', framealpha=0.8,
    facecolor='inherit', edgecolor='0.8',
    labelspacing=0.8, handlelength=2, handleheight=0.7,
):
    """ Hilbert feature analysis.

    author: Baolian shan <baolianshan@tju.edu.cn>
    Created on: 2023-03-11
    update log:
        None

    Parameters
    ----------
    times : ndarray
        Array of segment times.
    sfreq : int
        The sampling rate.
    signal : 1d-ndarray
        The orignal signal.
    envelope : 1d-ndarray
        The amplitude envelope.
    frequency : 1d-ndarray
        The instant frequency.
    phase : 1d-ndarray
        The instant phase.
    title : str
        The title of the time-frequency graph.
    signal_label、envelope_label、phase_label、frequency_label、axis_label : str
        The label of the orignal signal (amplitude envelope, instant frequency,
        instant phase、axis).
    signal_color、envelope_color、phase_color、frequency_color、axis_color : str
        The line color of the orignal signal
        (amplitude envelope, instant frequency,instant phase).
    fontsize : float or str
        If the font size is str, supported values are 'xx-small', 'x-small',
        'small', 'medium', 'large', 'x-large', 'xx-large'.
    fontweight : numeric value in range 0-1000 or str
        If the font weight is str, supported values are 'ultralight',
        'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
    fontcolor : str
        The font color, by default 'black', supported values are 'blue',
        'purple', 'orange', 'green', 'red', 'yellow', 'pink', 'lightgreen',
        'forestgreen', 'cyan', 'teal', 'gold', 'gray', 'olivedrab', 'sage'.
    fontstyle : str
        The font style, supported values are 'normal', 'italic', 'oblique'.
    linewidth : float
        The line width.
    linestyle : str
        The line style, by default 'solid', supported values are 'dashed',
        'dash-dot', 'dotted'.
    bbox_to_anchor : 2-tuple, or 4-tuple of floats
        The coordinate position of the legend.
    labelcolor : str or list
        The color of the text in the legend.
    framealpha : float
        The alpha transparency of the legend's background.
    facecolor : str
        The legend's background color, by default 'inherit'.
    edgecolor : str
        The legend's background patch edge color, by default '0.8'.
    labelspacing : float
        The vertical space between the legend entries, in font-size units.
    handlelength : float
        The length of the legend handles, in font-size units.
    handleheight : float
        The height of the legend handles, in font-size units.

    """
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    fig.suptitle(
        title,
        fontsize=fontsize_title,
        fontweight=fontweight_title,
        color=fontcolor_title,
        fontstyle=fontstyle_title)
    ax0.plot(
        times,
        signal,
        label=signal_label,
        color=signal_color,
        linewidth=linewidth,
        linestyle=linestyle)
    ax0.plot(
        times,
        envelope,
        label=envelope_label,
        color=envelope_color,
        linewidth=linewidth,
        linestyle=linestyle)
    ax0.set_xlabel(
        xlabel=axis_label,
        fontsize=fontsize,
        fontweight=fontweight,
        color=fontcolor,
        fontstyle=fontstyle)

    ax1.plot(times[1:],
             frequency,
             label=frequency_label,
             color=frequency_color,
             linewidth=linewidth,
             linestyle=linestyle)
    ax1.set_xlabel(
        xlabel=axis_label,
        fontsize=fontsize,
        fontweight=fontweight,
        color=fontcolor,
        fontstyle=fontstyle)
    ax1.set_ylim(-sfreq / 2, sfreq / 2)

    ax2.plot(
        times,
        phase,
        label=phase_label,
        color=phase_color,
        linewidth=linewidth,
        linestyle=linestyle)
    ax2.set_xlabel(
        xlabel=axis_label,
        fontsize=fontsize,
        fontweight=fontweight,
        color=fontcolor,
        fontstyle=fontstyle)
    ax2.set_ylim(-2 * np.pi, 2 * np.pi)
    fig.legend(bbox_to_anchor=bbox_to_anchor, fontsize=fontsize,
               labelcolor=labelcolor, framealpha=framealpha,
               facecolor=facecolor, edgecolor=edgecolor,
               labelspacing=labelspacing,
               handlelength=handlelength,
               handleheight=handleheight)
    fig.tight_layout()
    return


class CWT(BaseEstimator, TransformerMixin):
    """
    continuous wavelet transform (CWT).
    We apply CWT after data pre-processing in the version,
    to perform time-frequency analysis of the EEG signal.
    Performs a continuous wavelet transform on data,
    using the wavelet function. A CWT performs a convolution
    with data using the wavelet function,
    which is characterized by a width parameter and length parameter.
    The wavelet function is allowed to be complex.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-02-24

    update log:
        None

    Parameters
    ----------
    fs : int
        Sampling rate.
    wavelet : function
        Wavelet function, supported functions are 'morlet' or 'ricker'.
    freqs : 1d-array
        Frequency series of time-frequency analysis.
    omega0 : int
        The parameter related to the center frequency, by default '5'
    n_samples : int
        Maximum threshold of time window for time-frequency analysis.
        When the default is None, it is equal to the total number
        of sample points per trial.
    dtype : data-type, optional
        The desired data type of output. Defaults to float64 if the
        output of wavelet is real
        and complex128 if it is complex.

    # the variables of analysis:
    trail_id : int
        The index of the specified trial.
    channel_id : int
        The index of the specified channel.
    cmap : str or colormap
        The Colormap instance or registered colormap name used to
        map scalar data to colors, by default 'viridis',
        supported values are 'summer',
        'GnBu_r', 'RdPu_r', 'YlGn_r'......
    vmin, vmax : float, optional
        When using scalar data and no explicit norm,
        vmin and vmax define the data range that the colormap covers.
        By default 'None', the colormap covers the complete value
        range of the supplied data.
    title : str
        The title of the time-frequency graph.
    distance : float
        The horizontal distance between the time-frequency graph
        and the title of colorbar.


    Raises
    ----------
    ValueError
        None


    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using CWT

        from metabci.brainda.algorithms.feature_analysis. \
        time_frequency_analysis import CWT
        cwt = CWT(n_jobs=None, fs=srate, wavelet=signal.morlet2,
                freqs=freq, omega0=5,
                dtype="complex128", trail_id=0,
                channel_id=channel_id_class0,
                cmap='summer', vmin=None, vmax=None,
                fontsize_title=12, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='oblique',
                title="Class0: Complex Morlet Wavelet Transform of
                'C3'", distance=0.1)
        cwt.fit(X_class0)
        CWT_complexmatrix_class0, \
        CWT_spectrum_energy_class0 = cwt.transform(X_class0)
        cwt.draw()


    Note
    ----
    Font format (fontsize、fontweight、fontcolor、fontstyle...) can be set
    according to your needs.


    """

    def __init__(self, n_jobs=None, fs=None,
                 wavelet=signal.morlet2, freqs=None, omega0=5,
                 n_samples=None, dtype="complex128",
                 trail_id=1, channel_id=1,
                 cmap='viridis', vmin=None,
                 vmax=None, fontsize_title=14,
                 fontweight_title='bold',
                 fontcolor_title='limegreen',
                 fontstyle_title='oblique',
                 title="Complex Morlet Wavelet Transform",
                 distance=0.05):

        self.n_jobs = n_jobs
        self.fs = fs
        self.wavelet = wavelet
        self.freqs = freqs
        self.n_samples = n_samples
        self.dtype = dtype
        # the variables of analysis:
        self.trail_id = trail_id
        self.channel_id = channel_id
        self.omega0 = omega0
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.title = title
        self.distance = distance

    def fit(self, X: ndarray):
        # X.shape: (n_trials, n_channels, n_samples)
        self.n_channels = X.shape[1]
        if self.n_samples is None:
            self.n_samples = X.shape[2]
        return self

    def transform(self, X: ndarray):
        X = signal.detrend(X, axis=-1, type="linear")
        n_trails = X.shape[0]
        self.CWT_complexmatrix = np.zeros(
            (n_trails, self.n_channels, len(
                self.freqs), self.n_samples), dtype=self.dtype)
        # the relationship between width and frequency is as follows:
        widths = self.omega0 * self.fs / (2 * self.freqs * np.pi)
        for trial in range(n_trails):
            for channel in range(self.n_channels):
                self.CWT_complexmatrix[trial, channel, :, :] = signal.cwt(
                    X[trial, channel, :self.n_samples],
                    self.wavelet,
                    widths,
                    self.dtype,
                )
        self.CWT_spectrum_energy = np.abs(self.CWT_complexmatrix) ** 2
        return self.CWT_complexmatrix, self.CWT_spectrum_energy

    def draw(self):
        t_index = np.linspace(
            0,
            self.n_samples /
            self.fs,
            num=self.n_samples,
            endpoint=False)
        spectplot(t_index, self.freqs,
                  self.CWT_spectrum_energy[self.trail_id,
                                           self.channel_id, :, :],
                  cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                  title_label=self.title, distance=self.distance,
                  fontsize_title=self.fontsize_title,
                  fontweight_title=self.fontweight_title,
                  fontcolor_title=self.fontcolor_title,
                  fontstyle_title=self.fontstyle_title)
        return


class STFT(BaseEstimator, TransformerMixin):
    """
    short time Fourier transform (STFT).
    We apply STFT after data pre-processing in the version,
    to perform time-frequency analysis of the EEG signal.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-03-02

    update log:
        None


    Parameters
    ----------
    fs : int
        Sampling rate.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple,
        it is passed to `get_window` to generate the window values,
        which are DFT-even by default. See `get_window` for a list of
        windows and required parameters. If `window` is array_like
        it will be used directly as the window and its length must
        be nperseg. Defaults to a Hann window.
    n_samples : int
        Maximum threshold of time window for time-frequency analysis.
        When the default is None, it is equal to the total number of
        sample points per trial.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
        When specified, the COLA constraint must be met
        (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.
        If `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment.
        If `detrend` is a string,
        it is passed as the `type` argument to the `detrend` function.
        If it is a function,
        it takes a segment and returns a detrended segment.
        If `detrend` is `False`, no detrending is done.
        Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data.
        If `False` return a two-sided spectrum.
        Defaults to `True`, but for complex data,
        a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends,
        and how to generate the new values, in order to
        center the first windowed segment on the first input point.
        This has the benefit of enabling reconstruction of
        the first input point when the employed window function starts at zero.
        Valid options are ``['even', 'odd', 'constant', 'zeros', None]``.
        Defaults to 'zeros', for zero padding extension.
        I.e. ``[1, 2, 3, 4]`` is extended to ``[0, 1, 2, 3, 4, 0]``
        for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded
        at the end to make the signal fit exactly into an integer
        number of window segments, so that all of the signal is
        included in the output. Defaults to `True`.
        Padding occurs after boundary extension,
        if `boundary` is not `None`,
        and `padded` is `True`, as is the default.
    axis : int, optional
        Axis along which the STFT is computed; the default
        is over the last axis (i.e. ``axis=-1``).

    # the variables of analysis:
    trail_id : int
        The index of the specified trial.
    channel_id : int
        The index of the specified channel.
    cmap : str or colormap
        The Colormap instance or registered colormap name used to
        map scalar data to colors, by default 'viridis',
        supported values are 'summer', 'GnBu_r',
        'RdPu_r', 'YlGn_r'......
    vmin, vmax : float, optional
        When using scalar data and no explicit norm, vmin and vmax define the
        data range that the colormap covers. By default 'None',
        the colormap covers the complete value range of the supplied data.
    title : str
        The title of the time-frequency graph.
    distance : float
        The horizontal distance between the time-frequency graph and the
        title of colorbar.


    Raises
    ----------
    ValueError
        None


    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using STFT

        from metabci.brainda.algorithms.feature_analysis. \
        time_frequency_analysis import STFT
        stft = STFT(n_jobs=None, fs=srate, nperseg=64, nfft=X.shape[2],
                trail_id=20, channel_id=14, cmap='GnBu_r', vmin=None,
                vmax=None, fontsize_title=12, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='oblique',
                title="Trial 20-Channel 14: Short Time Fourier Transform",
                distance=0.25)
        stft.fit(X)
        STFT_complexmatrix, STFT_spectrum_energy = stft.transform(X)
        stft.draw()


    Note
    ----
    Font format (fontsize、fontweight、fontcolor、fontstyle...)
    can be set according to your needs.

    """

    def __init__(self, n_jobs=None, fs=None,
                 window='hann', n_samples=None,
                 nperseg=None, noverlap=None,
                 nfft=None, detrend=False,
                 return_onesided=True, boundary='zeros',
                 padded=True, axis=-1,
                 trail_id=1, channel_id=1,
                 cmap='viridis', vmin=None,
                 vmax=None, fontsize_title=14,
                 fontweight_title='bold', fontcolor_title='limegreen',
                 fontstyle_title='oblique',
                 title="Short Time Fourier Transform",
                 distance=0.05):
        self.n_jobs = n_jobs
        self.fs = fs
        self.window = window
        self.n_samples = n_samples
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.boundary = boundary
        self.padded = padded
        self.axis = axis
        # the variables of analysis:
        self.trail_id = trail_id
        self.channel_id = channel_id
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.title = title
        self.distance = distance

    def fit(self, X: ndarray):
        # X.shape: (n_trials, n_channels, n_samples)
        self.n_channels = X.shape[1]
        if self.n_samples is None:
            self.n_samples = X.shape[2]
        return self

    def transform(self, X: ndarray):
        X = signal.detrend(X, axis=-1, type="linear")
        n_trails = X.shape[0]
        self.data = signal.stft(X[0, 0, :self.n_samples], self.fs,
                                self.window, self.nperseg, self.noverlap,
                                self.nfft, self.detrend,
                                self.return_onesided, self.boundary,
                                self.padded, self.axis)
        # data is a tuple, data[0]=f, data[1]=t, data[2]=Zxx(needed complex
        # matrix)
        self.STFT_complexmatrix = np.zeros(
            (n_trails, self.n_channels,
             self.data[2].shape[0],
             self.data[2].shape[1]),
            dtype="complex128"
        )
        for trial in range(n_trails):
            for channel in range(self.n_channels):
                self.data = []
                self.data = signal.stft(X[trial, channel, :self.n_samples],
                                        self.fs, self.window, self.nperseg,
                                        self.noverlap, self.nfft,
                                        self.detrend, self.return_onesided,
                                        self.boundary, self.padded, self.axis)
                self.STFT_complexmatrix[trial, channel, :, :] = self.data[2]

        self.STFT_spectrum_energy = np.abs(self.STFT_complexmatrix) ** 2
        return self.STFT_complexmatrix, self.STFT_spectrum_energy

    def draw(self):
        spectplot(self.data[1], self.data[0],
                  self.STFT_spectrum_energy[
                      self.trail_id, self.channel_id, :, :
                    ],
                  cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                  title_label=self.title, distance=self.distance,
                  fontsize_title=self.fontsize_title,
                  fontweight_title=self.fontweight_title,
                  fontcolor_title=self.fontcolor_title,
                  fontstyle_title=self.fontstyle_title)
        return


class HT(BaseEstimator, TransformerMixin):
    """
    Hilbert transform(HT).
    We apply HT after data pre-processing in the version,
    to analyze the analytic signal, amplitude envelope,
    instant frequency, instant phase of the orignal EEG.

    author: Baolian shan <baolianshan@tju.edu.cn>

    Created on: 2023-03-11

    update log:
        None

    Parameters
    ----------
    fs : int
        Sampling rate.
    N : int
        Number of Fourier components. Default: ``x.shape[axis]``
    axis : int
        Axis along which to do the transformation. Default: -1.
    n_samples : int
        Maximum threshold of time window for time-frequency analysis.
        When the default is None, it is equal to the total number of
        sample points per trial.

    # the variables of analysis:
    trail_id : int
        The index of the specified trial.
    channel_id : int
        The index of the specified channel.
    title : str
        The title of the time-frequency graph.
    signal_label、envelope_label、phase_label、frequency_label、axis_label : str
        The label of the orignal signal (amplitude envelope, instant frequency,
        instant phase、axis).
    signal_color、envelope_color、phase_color、frequency_color、axis_color : str
        The line color of the orignal signal (amplitude envelope,
        instant frequency, instant phase).
    linewidth : float
        The line width.
    linestyle : str
        The line style, by default 'solid', supported values are 'dashed',
        'dash-dot', 'dotted'.
    bbox_to_anchor : 2-tuple, or 4-tuple of floats
        The coordinate position of the legend.
    labelcolor : str or list
        The color of the text in the legend.
    framealpha : float
        The alpha transparency of the legend's background.
    facecolor : str
        The legend's background color, by default 'inherit'.
    edgecolor : str
        The legend's background patch edge color, by default '0.8'.
    labelspacing : float
        The vertical space between the legend entries, in font-size units.
    handlelength : float
        The length of the legend handles, in font-size units.
    handleheight : float
        The height of the legend handles, in font-size units.


    Raises
    ----------
    ValueError
        None


    Tip
    ----
    .. code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: An example using HT

        from metabci.brainda.algorithms.feature_analysis.
        time_frequency_analysis import HT
        ht = HT(n_jobs=None, fs=srate, axis=-1,
                trail_id=20, channel_id=14,
                title="Trial 20-Channel 14: Hilbert Transform",
                fontsize_title=14, fontweight_title='bold',
                fontcolor_title='black', fontstyle_title='oblique',
                signal_label='orignal signal',
                envelope_label='amplitude envelope',
                phase_label='instantaneous phase',
                frequency_label='instantaneous frequency',
                axis_label='Time(s)',
                signal_color='blue', envelope_color='orange',
                frequency_color='red', phase_color='green',
                fontsize=10, fontweight='bold',
                fontcolor='black', fontstyle='normal',
                linewidth=2, linestyle='solid',
                bbox_to_anchor=(0.99,0.38), labelcolor='black',
                framealpha=0.7, facecolor='white', edgecolor='0.8',
                labelspacing=0.5, handlelength=2, handleheight=0.7)
        ht.fit(X)
        HT_complexmatrix, amplitude_envelope, instant_phase,
        instant_frequency = ht.transform(X)
        ht.draw()


    Note
    ----
    Font format (fontsize、fontweight、fontcolor、fontstyle...) can be set
    according to your needs.

    """

    def __init__(self, n_jobs=None, fs=None, N=None, axis=-1, n_samples=None,
                 trail_id=1, channel_id=1, title="Hilbert Transform",
                 fontsize_title=12, fontweight_title='bold',
                 fontcolor_title='black', fontstyle_title='oblique',
                 signal_label='orignal signal',
                 envelope_label='amplitude envelope',
                 phase_label='instantaneous phase',
                 frequency_label='instantaneous frequency',
                 axis_label='Time(s)', signal_color='blue',
                 envelope_color='orange', frequency_color='red',
                 phase_color='green', fontsize=10, fontweight='bold',
                 fontcolor='black', fontstyle='normal', linewidth=1,
                 linestyle='solid', bbox_to_anchor=(0.88, 0.33),
                 labelcolor='black', framealpha=0.8,
                 facecolor='inherit', edgecolor='0.8',
                 labelspacing=0.8, handlelength=2, handleheight=0.7):
        self.n_jobs = n_jobs
        self.fs = fs
        self.N = N
        self.axis = axis
        self.n_samples = n_samples
        # the variables of analysis:
        self.trail_id = trail_id
        self.channel_id = channel_id
        self.title = title
        self.fontsize_title = fontsize_title
        self.fontweight_title = fontweight_title
        self.fontcolor_title = fontcolor_title
        self.fontstyle_title = fontstyle_title
        self.signal_label = signal_label
        self.envelope_label = envelope_label
        self.phase_label = phase_label
        self.frequency_label = frequency_label
        self.axis_label = axis_label

        self.signal_color = signal_color
        self.envelope_color = envelope_color
        self.frequency_color = frequency_color
        self.phase_color = phase_color

        self.fontsize = fontsize
        self.fontweight = fontweight
        self.fontcolor = fontcolor
        self.fontstyle = fontstyle

        self.linewidth = linewidth
        self.linestyle = linestyle

        self.bbox_to_anchor = bbox_to_anchor
        self.labelcolor = labelcolor
        self.framealpha = framealpha
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.labelspacing = labelspacing
        self.handlelength = handlelength
        self.handleheight = handleheight

    def fit(self, X: ndarray):
        # X.shape: (n_trials, n_channels, n_samples)
        self.n_channels = X.shape[1]
        if self.n_samples is None:
            self.n_samples = X.shape[2]
        return self

    def transform(self, X: ndarray):
        n_trails = X.shape[0]
        self.HT_complexmatrix = np.zeros(
            (n_trails, self.n_channels, self.n_samples),
            dtype="complex128")
        self.amplitude_envelope = np.zeros(
            (n_trails, self.n_channels, self.n_samples))
        self.instant_phase = np.zeros(
            (n_trails, self.n_channels, self.n_samples))
        self.instant_frequency = np.zeros(
            (n_trails, self.n_channels, self.n_samples - 1))
        for trial in range(n_trails):
            for channel in range(self.n_channels):
                data = []
                data = signal.hilbert(X[trial, channel, :], self.N, self.axis)
                self.HT_complexmatrix[trial, channel, :] = data
                self.amplitude_envelope[trial, channel, :] = np.abs(data)
                self.instant_phase[trial, channel, :] = np.angle(data)
                self.instant_frequency[trial, channel, :] = (
                    np.diff(
                        np.unwrap(self.instant_phase[trial, channel, :])
                    ) / (2.0 * np.pi) * self.fs
                )
        return (
            self.HT_complexmatrix,
            self.amplitude_envelope,
            self.instant_phase,
            self.instant_frequency,
        )

    def draw(self):
        t = np.arange(self.n_samples) / self.fs
        _htplot(
            t,
            self.fs,
            np.real(self.HT_complexmatrix[self.trail_id, self.channel_id, :]),
            self.amplitude_envelope[self.trail_id, self.channel_id, :],
            self.instant_frequency[self.trail_id, self.channel_id, :],
            self.instant_phase[self.trail_id, self.channel_id, :],
            self.title,
            self.fontsize_title,
            self.fontweight_title,
            self.fontcolor_title,
            self.fontstyle_title,
            self.signal_label,
            self.envelope_label,
            self.phase_label,
            self.frequency_label,
            self.axis_label,
            self.signal_color,
            self.envelope_color,
            self.frequency_color,
            self.phase_color,
            self.fontsize,
            self.fontweight,
            self.fontcolor,
            self.fontstyle,
            self.linewidth,
            self.linestyle,
            self.bbox_to_anchor,
            self.labelcolor,
            self.framealpha,
            self.facecolor,
            self.edgecolor,
            self.labelspacing,
            self.handlelength,
            self.handleheight,
        )
        return
