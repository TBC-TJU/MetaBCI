# -*- coding: utf-8 -*-
# License: MIT License
"""
P300 offline analysis.

"""
import matplotlib.pyplot as plt
from mne.filter import filter_data
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import mne
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# global
global biglab, n_row, n_col, n_char
biglab = np.array([[21, 27, 33, 39, 45, 51],
                   [22, 28, 34, 40, 46, 52],
                   [23, 29, 35, 41, 47, 53],
                   [24, 30, 36, 42, 48, 54],
                   [25, 31, 37, 43, 49, 55],
                   [26, 32, 38, 44, 50, 56]])
n_row = 6
n_col = 6
n_char = 36  # char num


def find_encoding(label):
    """Find the index corresponding to label_big
    -author: Ruixin Luo
    -Created on: 2022-12-06
    -update log:
        None
    Parameters
    ----------
    label: big label in  biglab , int

    Returns
    -------
    index: the row and column index corresponding to label_big
           row_index, column_index
    """
    ind = np.where(biglab == label)
    row_ind = ind[0][0] + 1
    column_ind = ind[1][0] + 1 + n_row

    return row_ind, column_ind


def P300read_data(run_files, chs, interval, row_plus_col, n_rounds=6):
    """read P300 data.
    -author: Ruixin Luo
    -Created on: 2022-12-06
    -update log:
        None
    Parameters
    ----------
    run_files:  File path           , list
    chs:   Selected channels name   , list
    interval: time interval         , list[start, end]
    row_plus_col: n_rows + n_col    , int
    n_rounds:  the number of rounds , int

    Returns
    -------
    epochs_all: Extracted data for each command, shape(n_command * n_cnts,
    channels, n_times)
    label_id_all:  ID of all events,             shape(n_commands * n_cnts,
    n_events of singel command)
    label_loc_all: latency of all events,        shape(n_commands * n_cnts,
    n_events of singel command)
    """
    epochs_all = []
    label_id_all = []
    label_loc_all = []
    for run_file in run_files:  # different cnt
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw.set_eeg_reference(['M1', 'M2'])
        raw = upper_ch_names(raw)
        events, events_id = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False)
        # the index of pick channels
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        # Re-arrange events
        label_id = events[:, -1]  # ID of all events,shape(n_events,)
        label_loc = events[:,  0]  # Location of all events,shape(n_events,)
        # shape(n_commands,n_events of singel command) e.g, 36 * 73
        label_id = label_id.reshape(-1, (row_plus_col*n_rounds+1))
        # shape(n_commands,n_events of singel command) e.g, 36 * 73
        label_loc = label_loc.reshape(-1, (row_plus_col * n_rounds + 1))
        # latency: label_loc - base_line(the biglab: label_loc[:, 0])
        label_loc = label_loc - \
            np.expand_dims(label_loc[:, 0], 1).repeat(
                label_loc.shape[1], axis=1)
        # Extracting data for each command
        biglab_id = label_id[:, 0]
        for label in biglab_id:
            # print("label=",label)
            epochs = mne.Epochs(raw,
                                events,
                                event_id=label,
                                tmin=interval[0],
                                tmax=interval[1],
                                baseline=None,
                                picks=ch_picks,
                                verbose=False).get_data() * 1e6
            epochs_all.append(epochs)
        label_id_all.append(label_id)
        label_loc_all.append(label_loc)
    # concatenate
    # (n_commands * n_cnts) * channels * n_times   e.g. (36*2) * 6 * 1500
    epochs_all = np.concatenate(epochs_all, 0)
    # shape(n_commands * n_cnts ,n_events of singel command) e.g, (36*2) * 73
    label_id_all = np.concatenate(label_id_all, 0)
    # shape(n_commands * n_cnts ,n_events of singel command) e.g, (36*2) * 73
    label_loc_all = np.concatenate(label_loc_all, 0)
    return epochs_all, label_id_all, label_loc_all, ch_picks


def train_model(epochs_all,
                label_id_all,
                label_loc_all,
                signal_time=0.5,
                fs=1000,
                Analysis_down=1):
    """train P300 model.
    -author: Ruixin Luo
    -Created on: 2022-12-06
    -update log:
        None
    Parameters
    ----------
    epochs_all: Extracted data for each command,    shape(n_command * n_cnts,
    channels, n_times)
    label_id_all:  ID of all events,                shape(n_commands * n_cnts,
    n_events of singel command)
    label_loc_all: latency of all events,           shape(n_commands * n_cnts,
    n_events of singel command)
    fs: the sampling rate                           int
    f_down: the down-sampling rate                  int
    Analysis_down : In LDA, down-sampling interval  int


    Returns
    -------
    P300_model
    """
    # Filtering, all epoch
    epochs_all = filter_data(epochs_all, sfreq=fs,
                             l_freq=1, h_freq=20, n_jobs=4, method='fir')
    n_commands = label_id_all.shape[0]

    # Finding targets and non-targets for every command
    targets = []
    notargets = []
    for i in range(n_commands):
        epochs = epochs_all[i, ...]
        label_big = label_id_all[i, 0]
        label_small = label_id_all[i, 1:]
        label_latency = label_loc_all[i, 1:]
        # Find the index corresponding to label_big
        row_ind, column_ind = find_encoding(label=label_big)
        target_ind = np.where((label_small == row_ind) | (
            label_small == column_ind))[0]  # shape(2 * n_rounds,)
        notarget_ind = np.where((label_small != row_ind) & (
            label_small != column_ind))[0]  # shape(10 * n_rounds,)
        # Find target data
        for i_tar in target_ind:
            latency1 = label_latency[i_tar]
            a_target = epochs[:, int(latency1):int(latency1+signal_time*fs)]
            targets.append(np.expand_dims(a_target, 0))
        # Find non-target data
        for i_notar in notarget_ind:
            latency2 = label_latency[i_notar]
            a_notarget = epochs[:, int(latency2):int(
                latency2 + signal_time * fs)]
            notargets.append(np.expand_dims(a_notarget, 0))
    # shape(2 * n_rounds *n_commands, n_channels, n_times)  e.g, 864, 6, 750
    targets = np.concatenate(targets)
    # shape(10 * n_rounds *n_commands, n_channels, n_times)  e.g, 4320, 6, 750
    notargets = np.concatenate(notargets)

    #  down-sampling
    targets = targets[:, :, 0:targets.shape[2]:Analysis_down]
    notargets = notargets[:, :, 0:notargets.shape[2]:Analysis_down]

    # train
    Xtrain = np.concatenate((targets, notargets), axis=0)
    [all_trails, chans, sample_num] = Xtrain.shape
    XtrainS2 = Xtrain.reshape(all_trails, chans * sample_num)
    label_tar = np.ones([targets.shape[0]])
    label_nontar = np.zeros([notargets.shape[0]])
    Xlabel = np.concatenate((label_tar, label_nontar), axis=0)
    LDA_model = LDA()
    LDA_model.fit(XtrainS2, Xlabel)

    return LDA_model


def Time_Analysis(epochs_all,
                  label_id_all,
                  label_loc_all,
                  baseline_time=0.3,
                  signal_time=0.5,
                  fs=1000,
                  Is_draw=True,
                  channel_draw=1):
    """Time_Analysis for P300.
    -author: Ruixin Luo
    -Created on: 2022-12-06
    -update log:
        None
    Parameters
    ----------
    epochs_all: Extracted data for each command, shape(n_command * n_cnts,
    channels, n_times)
    label_id_all:  ID of all events,             shape(n_commands * n_cnts,
    n_events of singel command)
    label_loc_all: latency of all events,        shape(n_commands * n_cnts,
    n_events of singel command)
    fs: the sampling rate                        int
    Is_draw: plot target and no-target           bool
    channel_draw: the index of channel to draw   int

    Returns
    -------
    """
    # Filtering, all epoch
    epochs_all = filter_data(epochs_all, sfreq=fs,
                             l_freq=1, h_freq=10, n_jobs=4, method='fir')
    n_commands = label_id_all.shape[0]

    # Finding targets and non-targets for every command
    targets = []
    notargets = []
    for i in range(n_commands):
        epochs = epochs_all[i, ...]
        label_big = label_id_all[i, 0]
        label_small = label_id_all[i, 1:]
        label_latency = label_loc_all[i, 1:]
        # Find the index corresponding to label_big
        row_ind, column_ind = find_encoding(label=label_big)
        target_ind = np.where((label_small == row_ind) | (
            label_small == column_ind))[0]  # shape(2 * n_rounds,)
        notarget_ind = np.where((label_small != row_ind) & (
            label_small != column_ind))[0]  # shape(10 * n_rounds,)
        # Find target data
        for i_tar in target_ind:
            latency1 = label_latency[i_tar]
            a_target = epochs[:, int(
                latency1-baseline_time*fs):int(latency1+signal_time*fs)]
            targets.append(np.expand_dims(a_target, 0))
        # Find non-target data
        for i_notar in notarget_ind:
            latency2 = label_latency[i_notar]
            a_notarget = epochs[:, int(
                latency2-baseline_time*fs):int(latency2 + signal_time * fs)]
            notargets.append(np.expand_dims(a_notarget, 0))
    # shape(2 * n_rounds *n_commands, n_channels, n_times)  e.g, 864, 6, 750
    targets = np.concatenate(targets)
    # shape(10 * n_rounds *n_commands, n_channels, n_times)  e.g, 4320, 6, 750
    notargets = np.concatenate(notargets)

    # Plotting ERP waveforms
    if Is_draw is True:
        mean_target = np.mean(targets, 0)[channel_draw, :]
        mean_notarget = np.mean(notargets, 0)[channel_draw, :]
        # 去基线
        base_tar = np.mean(mean_target[:int(fs * baseline_time)])
        base_no_tar = np.mean(mean_notarget[:int(fs * baseline_time)])
        mean_target = mean_target - base_tar
        mean_notarget = mean_notarget - base_no_tar

        # 绘图
        t = np.arange(-baseline_time, signal_time, 1 / fs)
        plt.plot(t, mean_target, label='mean_target')
        plt.plot(t, mean_notarget, label='mean_notarget')
        plt.plot(t, mean_target-mean_notarget, label='diff')
        plt.legend()
        plt.show()

# LDA——predict


def LDA_predict(LDA_model,
                epoch_test,
                label_loc_test,
                label_id_test,
                stim_round,
                row_plus_col,
                signal_time=0.75,
                fs=1000,
                Analysis_down=1):
    ''' Train model and predict label
    -author: Shengfu Wen
    -Created on: 2022-12-06
    -update log:
    Parameters
    ----------
    LDA_model    : LDA model
    epoch_test   : a long time data, shape( chan, sample)
    label_loc_test : label latency, shape( row_plus_col * n_rounds)
    label_id_test :label id, shape( row_plus_col * n_rounds)
    stim_round   : stimulation round of  a single command      int
    row_plue_col : row plus col                                int
    fs: the sampling rate                                      int
    f_down: the down-sampling rate                             int
    Analysis_down : In LDA, down-sampling interval             int

    Returns
    -------
    predict_lab : predict label                                int
    '''

    # prepare test data
    # filter
    epoch_test = filter_data(epoch_test, sfreq=fs,
                             l_freq=1, h_freq=20, n_jobs=4, method='fir')
    label_latency = label_loc_test

    # Intercept data by events from smallest to largest
    # shape(channel,n_times,row_plus_col,n_rounds)
    Xtest = np.zeros([epoch_test.shape[0], int(
        signal_time*fs), row_plus_col, stim_round])
    for i in range((row_plus_col)):  # i-th label
        # find the index of i-th event
        ind_event = np.where(label_id_test == i+1)[0]
        for n, n_ind in enumerate(ind_event):  # n-th round
            lat = label_latency[n_ind]
            dat = epoch_test[:, int(lat):int(lat + signal_time * fs)]
            Xtest[:, :, i, n] = dat
    # down_sampling
    Xtest = Xtest[:, 0: Xtest.shape[1]: Analysis_down, :, :]
    Xtest_r = Xtest.reshape(
        int(Xtest.shape[0]*Xtest.shape[1]), row_plus_col, stim_round)

    # Calculate the correlation coefficient
    rr = np.zeros([row_plus_col, stim_round])
    for m in range((row_plus_col)):
        for j in range((stim_round)):
            data = Xtest_r[:, m, j]
            rr[m, j] = np.dot(data, np.transpose(
                LDA_model.coef_)) + LDA_model.intercept_

    rr_mean = np.mean(rr, -1)  # mean all rounds
    row_max = np.argmax(rr_mean[0:n_row])
    col_max = np.argmax(rr_mean[n_row:n_col + n_row])

    predict_lab = biglab[row_max, col_max]
    return predict_lab


def offine_validation(train_run_files,
                      test_run_files,
                      pick_chs,
                      command_interval,
                      signal_time,
                      row_plus_col,
                      n_rounds,
                      Analysis_down=10):
    # train_model
    epochs_all, label_id_all, label_loc_all, _ = P300read_data(
        train_run_files,
        chs=pick_chs,
        interval=command_interval,
        row_plus_col=row_plus_col,
        n_rounds=n_rounds)
    model_p300 = train_model(epochs_all,
                             label_id_all,
                             label_loc_all,
                             signal_time=signal_time,
                             fs=srate,
                             Analysis_down=Analysis_down)

    # test
    epochs_test, label_id_test, label_loc_test, _ = P300read_data(
        test_run_files,
        chs=pick_chs,
        interval=command_interval,
        row_plus_col=row_plus_col,
        n_rounds=n_rounds)
    n_command = label_id_test.shape[0]
    true_num = 0
    TrueandPredict_label = np.zeros([int(n_command), 2])
    for com in range(n_command):
        epoch = epochs_test[com, ...]
        # shape((n_rounds * row_plus_col),)
        label_loc = label_loc_test[com, 1:]
        # shape((n_rounds * row_plus_col),)
        label_id = label_id_test[com, 1:]
        predict_label = LDA_predict(model_p300,
                                    epoch,
                                    label_loc,
                                    label_id,
                                    stim_round=n_rounds,
                                    row_plus_col=row_plus_col,
                                    signal_time=signal_time,
                                    fs=srate,
                                    Analysis_down=Analysis_down)
        TrueandPredict_label[com, 0] = label_id_test[com, 0]
        TrueandPredict_label[com, 1] = predict_label
        if label_id_test[com, 0] == predict_label:
            true_num += 1
    print('Acc:', true_num/n_command)
    print(TrueandPredict_label)
    return true_num/n_command


if __name__ == '__main__':
    # 初始化参数
    srate = 1000  #
    command_interval = [0, 15]  # 0.175 * 12 * 6 = 12.6
    stim_labels = list(range(1, 30))  # 事件标签
    pick_chs = ['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'OZ']  # 使用导联
    n_rounds = 6
    row_plus_col = 12  # 6 + 6
    Analysis_down = 10  # LDA down sampling
    filepath = "E:\\ShareFolder\\meta1207wsf\\P300\\train\\sub1"  # 数据的相对路径

    # Cross-validation
    n_cnts = 3
    n_cross = n_cnts
    acc_all = 0
    for i_cross in range((n_cross)):
        ind_all = list(range(1, n_cnts+1))
        ind_test = [i_cross + 1]
        ind_train = ind_all
        del ind_train[i_cross]
        print("ind_train=", ind_train)
        print("ind_test=", ind_test)
        train_run_files = ['{:s}\\{:d}.cnt'.format(
            filepath, run) for run in ind_train]    # train
        test_run_files = ['{:s}\\{:d}.cnt'.format(
            filepath, run) for run in ind_test]    # test
        acc = offine_validation(
            train_run_files=train_run_files,
            test_run_files=test_run_files,
            pick_chs=pick_chs,
            command_interval=command_interval,
            signal_time=0.75,
            row_plus_col=row_plus_col,
            n_rounds=n_rounds,
            Analysis_down=Analysis_down)
        acc_all = acc_all + acc

    # draw p300
    draw_runs = list(range(1, 4))  # 1,2
    draw_run_files = ['{:s}\\{:d}.cnt'.format(
        filepath, run) for run in draw_runs]    # train
    epochs_draw, label_id_draw, label_loc_draw, _ = P300read_data(
        draw_run_files,
        chs=pick_chs,
        interval=command_interval,
        row_plus_col=row_plus_col,
        n_rounds=n_rounds)
    Time_Analysis(epochs_draw,
                  label_id_draw,
                  label_loc_draw,
                  baseline_time=0.3,
                  signal_time=0.75,
                  fs=srate,
                  Is_draw=True,
                  channel_draw=1)

    # print acc
    print("final acc=", acc_all/n_cross)
