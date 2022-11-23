import numpy as np
from scipy.signal import detrend, stft
import mne
from scipy.signal import hilbert
import matplotlib.pyplot as plt


class TimeFrequencyAnalysis:
    def __init__(self, fs):
        self.fs = fs

    def func_morlet_wavelet(self, data, xtimes, omega, sigma, fs=None):
        """
        -author: Xin Fengran
        -Created on: 2022-8-8
        -update log:
            2022-8-11 by Xin Fengran
        Args:
            data: ndarray(Nchannel, Ntimes), the EEG data
            xtimes: ndarray(N,), timeline of the EEG data
            omega: float.
            sigma: float.
            fs: float, the resampling rate
        Returns:
            P: ndarray(Nchannel, N, nTimes), square amplitude of Morlet wavelet transform
            S: ndarray(Nchannel, N, nTimes), complex values of Morlet wavelet transform

        """
        data = detrend(data, axis=-1, type="linear")
        N_T = data.shape[1]
        N_C = data.shape[0]
        N_F = xtimes.shape[0]
        if fs is None:
            fs = self.fs
        f = xtimes / fs
        S = np.zeros((N_C, N_F, N_T))
        P = np.zeros((N_C, N_F, N_T))

        L_hw = N_T
        for fi in range(N_F):
            scaling_factor = omega / f[fi]
            u = (-np.arange(-L_hw, L_hw + 1)) / scaling_factor
            hw = (
                np.sqrt(1 / scaling_factor)
                * np.exp(-(u**2) / (2 * sigma**2))
                * np.exp(1j * 2 * np.pi * omega * u)
            )
            for ci in range(N_C):
                S_full = np.convolve(data[ci, :], hw.conjugate())
                S[ci, fi, :] = S_full[L_hw : L_hw + N_T]

        P = np.abs(S) ** 2
        return P, S

    def fun_stft(
        self,
        data,
        fs=None,
        window="hann",
        nperseg=256,
        noverlap=None,
        nfft=None,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        axis=-1,
    ):
        """
        -author: Xin Fengran
        -Created on: 2022-8-8
        -updata log:
            2022-8-11 by Xin Fengran
        Args:
            data(ndarray): the EEG data
            fs(float): the rasampling rate
            window(str or tuple or ndarray): desired window to use.
            nperseg(int): length of each segment.
            noverlap(int): number of points to overlap between segments.
            nfft(int): length of the FFT used.
            detrend(str or function or False): specifies how to detrend each segment.
            return_onesided(bool): If True, return a one-sided spectrum for real data. If False return a two-sided spectrum.
            Defaults to True, but for complex data, a two-sided spectrum is always returned.
            boundary(str): Specifies whether the input signal is extended at both ends, and how to generate the new values,
            in order to center the first windowed segment on the first input point.
            padded(bool): Specifies whether the input signal is zero-padded at the end.
            axis(int): axis along which the STFT is computed
        Returns:
            f(ndarray): array of sample frequencies
            t(ndarray): array of segment times
            Zxx(ndarray): the STFT of the EEG data
        """
        if fs is None:
            fs = self.fs
        f, t, Zxx = stft(
            data,
            fs,
            window,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            boundary,
            padded,
            axis,
        )
        return f, t, Zxx

    def fun_topoplot(self, X, chan_names, sfreq=None, ch_types="eeg"):
        """
        -author: Li Xiaoyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-11 by Li Xiaoyu

        Args:
            X(ndarray): the input data
            chan_names(list): the name of channels
            sfreq(float): the sampling rate
            ch_types(str): the type of channel

        """
        if sfreq is None:
            sfreq = self.fs

        montage_type = mne.channels.make_standard_montage(
            "standard_1020", head_size=0.1
        )
        epoch_info = mne.create_info(
            ch_names=chan_names, ch_types=ch_types, sfreq=sfreq
        )
        epoch_info.set_montage(montage_type)

        # The value specifying the upper bound of the color range.
        vmax = np.max(X)
        # The value specifying the lower bound of the color range.
        vmin = np.min(X)

        # plot
        fig, ax = plt.subplots(1, figsize=(4, 4), sharex=True, sharey=True)
        im, cn = mne.viz.plot_topomap(
            X, epoch_info, axes=ax, show=False, vmax=vmax, vmin=vmin, cnorm=None
        )
        ax.set_title("topomap", fontsize=25)
        plt.colorbar(im)
        plt.show()

    def fun_hilbert(self, X, N=None, axis=-1):
        """
        -author: Li Xiaoyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-11 by Li Xiaoyu

        Args:
            X(ndarray): the input data
            N(int): length of the hilbert used.
        Returns:
            analytic_signal(ndarray): discrete-time analytic signal
            realEnv(ndarray): the real part of the discrete-time analytic signal.
            imagEnv(ndarray): the imaginary part of the discrete-time analytical signal.
            angle(ndarray): the angle of the discrete-time analytical signal.
            envModu(ndarray): the envelope of the discrete-time analytical signal

        """
        analytic_signal = hilbert(X, N, axis)
        realEnv = np.real(analytic_signal)
        imagEnv = np.imag(analytic_signal)
        angle = np.angle(analytic_signal)
        envModu = np.sqrt(realEnv**2 + imagEnv**2)
        return analytic_signal, realEnv, imagEnv, angle, envModu
