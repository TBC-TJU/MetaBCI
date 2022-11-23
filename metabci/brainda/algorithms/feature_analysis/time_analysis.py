import numpy as np
import mne
import matplotlib.pyplot as plt


class TimeAnalysis:
    Chan_Neuroscan = [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "M1",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "M2",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "OZ",
        "O2",
        "CB2",
    ]
    Chan_Standard1020 = [
        "Fp1",
        "Fpz",
        "Fp2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "M1",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "M2",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POz",
        "PO4",
        "PO6",
        "PO8",
        "I1",
        "O1",
        "Oz",
        "O2",
        "I2",
    ]

    def __init__(self, data, meta, dataset, event, latency=0.0, channel=[0]):
        """
        -author: Wujieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray):the EEG data
            meta(dataframe): information of the selected data
            dataset(dataset): BaseDataset
            event(string): event trigger
            latency(float): start moment of the epoch, by default 0
            channel(list): selected channels, by default [0]
        """
        sub_meta = meta[meta["event"] == event]
        event_id = sub_meta.index.to_numpy()
        self.data_length = np.round(data.shape[2] / dataset.srate)

        if isinstance(channel[0], str):
            self.chan_ID = self.get_chan_id(channel, dataset.channels)
        elif isinstance(channel[0], int):
            self.chan_ID = channel
        self.data = data[event_id, :, :]
        self.latency = latency
        self.fs = dataset.srate
        self.All_channel = dataset.channels
        self.event = event

    def stacking_average(self, data=[], _axis=[0]):  # data：trials*channels*time
        """
        -author: Jiang Hanzhe
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Jiang Hanzhe
        Args:
            data(ndarray): the EEG data, by default []
            _axis(list): selected the dimensions, by default [0]
        Returns:
            data_mean(ndarray): array of averaged signals
        """
        if isinstance(_axis, int):
            _axis = [_axis]
        axis = tuple(_axis)
        if data == []:
            data = self.data
        data_mean = data.mean(axis=axis)
        return data_mean

    def peak_amplitude(self, data=[], time_start=0, time_end=1):
        """
        -author: Jiang Hanzhe
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Jiang Hanzhe
        Args:
            data(ndarray): the EEG data, by default []
            time_start(int): beginning of peak seeking, by default 0
            time_end(int): end of peak seeking, by default 1
        Returns:
            peak_amp(float): signal peak amplitude within the specified time quantum
        """
        if data == []:
            data1 = self.stacking_average()
            data = np.squeeze(data1[self.chan_ID, :])
        peak_amp = np.max(data[time_start:time_end])
        return peak_amp

    def average_amplitude(self, data=[], time_start=0, time_end=1):
        """
        -author: Jiang Hanzhe
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Jiang Hanzhe
        Args:
            data(ndarray): the EEG data, by default []
            time_start(int): beginning of peak seeking, by default 0
            time_end(int): end of peak seeking, by default 1
        Returns:
            ave_amp(float): signal average amplitude within the specified time quantum
        """
        if data == []:
            data1 = self.stacking_average()
            data = np.squeeze(data1[self.chan_ID, :])
        ave_amp = np.mean(data[time_start:time_end])
        return ave_amp

    def peak_latency(self, data=[], time_start=0, time_end=1):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray): the EEG data, by default []
            time_start(int): beginning of peak seeking, by default 0
            time_end(int): end of peak seeking, by default 1
        Returns:
            peak_loc(int): location of peak amplitude
            peak_amp(float): signal peak amplitude within the specified time quantum
        """
        if data == []:
            data1 = self.stacking_average()
            data = np.squeeze(data1[self.chan_ID, :])
        peak_amp = self.peak_amplitude(data, time_start, time_end)
        peak_loc = np.argmax(data[time_start:time_end]) + time_start
        return peak_loc, peak_amp

    def average_latency(self, data=[], time_start=0, time_end=1):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray): the EEG data, by default []
            time_start(int): beginning of peak seeking, by default 0
            time_end(int): end of peak seeking, by default 1
        Returns:
            ave_loc(int): location of average amplitude
            ave_amp(float): signal average amplitude within the specified time quantum
        """
        if data == []:
            data1 = self.stacking_average()
            data = np.squeeze(data1[self.chan_ID, :])
        ave_amp = self.average_amplitude(data, time_start, time_end)
        sample = time_end - time_start - 1
        half_average = np.sum(data) / 2
        integal = []
        for samp_i in range(sample):
            integal.append(np.abs(np.sum(data[0:samp_i]) - half_average))
        ave_loc = np.argmin(integal) + time_start

        return ave_loc, ave_amp

    def plot_single_trial(
        self, data, sample_num, axes=None, amp_mark=False, time_start=0, time_end=1
    ):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray): the EEG data
            sample_num(int): total number of sampling points in data
            axes(axessubplot): drawing area
            amp_mark(string): 'peak' or 'average', call different peak marking methods,
                               by default False (not marked)
            time_start(int): beginning of peak seeking, by default 0
            time_end(int): end of peak seeking, by default 1
        Returns:
            loc(int): location of amplitude
            amp(float): signal amplitude within the specified time quantum
            ax(axessubplot): drawing area
        """
        latency = self.latency
        data_mean = data
        import matplotlib.pyplot as plt

        fs = self.fs
        ax = axes if axes else plt.gca()
        t = np.arange(latency, sample_num / fs + latency, 1 / fs)
        plt.plot(t, data_mean)
        if amp_mark:
            func_str = "self." + amp_mark + "_latency"
            loc, amp = eval(func_str)(data_mean, time_start, time_end)
            plt.scatter(t[loc], amp, c="r", marker="o", label="振幅")
            pass
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        plt.xlabel("time[s]")
        plt.ylabel("amplitude[μV]")
        plt.legend()
        return loc, amp, ax

    def get_chan_id(self, ch_name, channels):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            ch_name(list): selected channels
            channels(list): standard channel list
        Returns:
            Chan_ID(list): index of ch_name in channels
        """
        Start = 0
        End = 64
        Chan_ID = []
        for i in range(len(ch_name)):
            Chan_ID.append(channels.index(ch_name[i].upper(), Start, End))
        pass
        return Chan_ID

    def plot_multi_trials(self, data, sample_num, axes=None):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray): the EEG data
            sample_num(int): total number of sampling points in data
            axes(axessubplot): drawing area
        Returns:
            ax(axessubplot): drawing area
        """
        data_shape = data.shape
        # data_sample = data_shape[1]
        trial_num = data_shape[0]
        fs = self.fs
        latency = self.latency
        t = np.arange(latency, sample_num / fs + latency, 1 / fs)
        ax = axes if axes else plt.gca()
        for trial_i in range(trial_num):
            plt.plot(t, data[trial_i, :], color=[0.7, 0.7, 0.7], linewidth=0.8)
        data_mean = np.mean(data, 0)
        plt.plot(t, data_mean, color=[1, 0, 0], linewidth=1.5, label="时域")
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        plt.legend()
        plt.xlabel("time[s]")
        plt.ylabel("amplitude[μV]")
        plt.legend(loc="lower right")
        return ax

    def plot_topomap(
        self, data, point, channels, fig, srate=-1, ch_types="eeg", axes=None
    ):
        """
        -author: Wu Jieyu
        -Created on: 2022-8-8
        -updata log:
            2022-8-15 by Wu Jieyu
        Args:
            data(ndarray): the EEG data
            point(int): selected sampling point
            channels(list): selected channels
            fig(figure): figure
            srate(int): sampling rate, by dafault self.fs
            ch_types(list of str | str): channel types, by default 'eeg'
            axes(axessubplot): drawing area, by default none
        Returns:
            aximage(axessubplot): drawing area
        """
        ch_names = []
        if srate == -1:
            srate = self.fs
        if isinstance(channels[0], str):
            Chan_ID = self.get_chan_id(channels, self.Chan_Neuroscan)
        elif isinstance(channels[0], int):
            Chan_ID = channels
        for i in Chan_ID:
            ch_names.append(self.Chan_Standard1020[i])

        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        evoked = mne.EvokedArray(data, info)
        evoked.set_montage("standard_1005")
        # ax = axes if axes else plt.gca()
        left = 0.92
        b = 0.2
        w = 0.015
        h = 0.6
        aximage, countour = mne.viz.plot_topomap(
            evoked.data[:, point],
            evoked.info,
            show=False,
            cmap="jet",
            vmin=data[:, point].min(),
            vmax=data[:, point].max(),
            contours=6,
        )
        rect = [left, b, w, h]
        cbar_ax = fig.add_axes(rect)
        plt.colorbar(aximage, cax=cbar_ax)

        return aximage
