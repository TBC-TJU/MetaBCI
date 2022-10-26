# -*- coding: utf-8 -*-
# License: MIT License
"""
P300 Feedback on NeuroScan.

"""
import numpy as np
import os
import time
import mne
from mne.filter import resample
from mne import Epochs
from scipy import signal
from numpy import linalg as LA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.io import concatenate_raws
from pylsl import StreamInfo, StreamOutlet

from metabci.brainflow.amplifiers import NeuroScan, Marker
from metabci.brainflow.workers import ProcessWorker

from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline


def chans_pick(rawdata, chans=None):
    """
    Parameters
    ----------
    rawdata : array, [chans_num, class_num, trial_num, sample_num]
    chans : list, channels name

    Returns
    -------
    data : array, [chans_num, class_num, trial_num, sample_num]
    """
    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
        'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
        'FC6', 'FC8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
        'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2',
        'CB2'
    ]

    idx_loc = []
    if isinstance(chans, list):
        for chans_value in chans:
            idx_loc.append(CHANNELS.index(chans_value.upper()))

    data = rawdata[idx_loc, ...] if idx_loc else rawdata
    return data


class MaxClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


def P300read_data(filepath):
    filelist = []
    for file in os.listdir(filepath):
        filefullpath = os.path.join(filepath, file)
        filelist.append(filefullpath)

    raw_cnts = []
    for file in filelist:
        print('loading %s ...' % file)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_cnt = mne.io.read_raw_cnt(file,
                                      eog=['HEO', 'VEO'],
                                      emg=['EMG'],
                                      ecg=['EKG'],
                                      preload=True,
                                      verbose=False)
        raw_cnts.append(raw_cnt)
    raw = concatenate_raws(raw_cnts)

    return raw


def P300train_predict_model(raw, n_char, row_plus_col, row, col, fs_p300,
                            stim_interval):
    events, events_id = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info,
                           emg=False,
                           eeg=True,
                           stim=False,
                           eog=False)
    picks_ch_names = [raw.ch_names[i] for i in picks]
    # trial_num
    trial_events = events[0:events.shape[0]:row_plus_col + 1, :]

    rawData = {}
    dataDic = {}
    for marker_i in range(0, row_plus_col):  # notice row/col labels
        print(marker_i + n_char)
        rawData[marker_i] = Epochs(
            raw,
            events=events,
            event_id=marker_i + 14,  # !!!!!! notice !!!!!!
            tmin=stim_interval[0],
            picks=picks,
            tmax=stim_interval[1],
            baseline=None,
            preload=True).get_data() * 1e6
        # filter
        # dataDic[marker_i] = filter_data(rawData[marker_i],
        #                                 sfreq=sfreq,
        #                                 l_freq=l_freq,
        #                                 h_freq=h_freq,
        #                                 n_jobs=4,
        #                                 method='fir')

        # downsampling / by 'down'
        dataDic[marker_i] = mne.filter.resample(rawData[marker_i],
                                                down=4,
                                                n_jobs=4)  # 1000->1000

    data = list(dataDic.values())
    # dataArray = data
    dataArray = np.array(
        data)  # stim_type(row plus col) * trials * channels * sample_num
    # np.save('./l_r/hejiatong', dataArray)

    ### data processing
    # heavy reference

    # channel selection

    rawdata = np.transpose(
        dataArray,
        [2, 0, 1, 3
         ])  # channels * stim_type(row plus col) * trials * sample_num
    channel_data = chans_pick(rawdata,
                              chans=['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'OZ'])

    # filter
    lpass = 1  # 低通截至频率
    hpass = 12  # 高通截至频率
    filterorder = 3  # 定义带通滤波器的阶数
    # fs_p300 = 1000  # 生成数字滤波器
    sos = signal.butter(filterorder, [lpass, hpass],
                        btype='band',
                        analog=False,
                        fs=fs_p300,
                        output='sos')
    fliter_p300 = signal.sosfiltfilt(
        sos, channel_data,
        axis=3)  # 6 channels * stim_type(row plus col) * trials * sample_num

    ### find template and non-template
    each_trial_tar = 2
    len_trials = trial_events.shape[0]
    [chann, stim_type, trials, sample_num] = fliter_p300.shape
    sample_num_s = len(fliter_p300[1, 1, 1, 0:sample_num:5])
    target_Temp = np.zeros([each_trial_tar * len_trials, chann, sample_num_s])
    nontarget_Temp = np.zeros([(row_plus_col - each_trial_tar) * len_trials,
                               chann, sample_num_s])
    event_labels = np.zeros([len_trials, col + row])
    event_labels_ch = np.zeros([len_trials, 1])
    # set event labels
    True_label = np.array([
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 3, 4, 5, 6, 7, 8, 9
    ])
    print(True_label)
    for eve in range(len_trials):
        event_lab = trial_events[eve, 2]  # event_lab:1-20
        event_labels_ch[eve, 0] = True_label[event_lab - 1]
        [col_id, row_id] = divmod(True_label[event_lab - 1], row)
        if row_id == 0:
            col_id = col_id - 1
            row_id = row
        event_labels[eve, row_id - 1] = 1
        event_labels[eve, row + col_id] = 1

    for tr in range(len_trials):
        tar = 0
        nontar = 0
        for rc in range(row_plus_col):
            if event_labels[tr, rc] == 1:
                target_Temp[tr * each_trial_tar +
                            tar, :, :] = fliter_p300[:, rc, tr, 0:sample_num:5]
                tar = tar + 1
            else:
                nontarget_Temp[tr * (row_plus_col - each_trial_tar) +
                               nontar, :, :] = fliter_p300[:, rc, tr,
                                                           0:sample_num:5]
                nontar = nontar + 1

    # change data size : trial *(chan * sample)
    target_Temp_S2 = target_Temp.reshape(each_trial_tar * len_trials,
                                         chann * sample_num_s)
    nontarget_Temp_S2 = nontarget_Temp.reshape(
        (row_plus_col - each_trial_tar) * len_trials, chann * sample_num_s)
    # np.save('./l_r/hjt', LDA_model)  # save template

    ###
    predict_labels = np.zeros([len_trials, col + row])
    true_num = 0
    true_row_num = 0
    true_col_num = 0
    for tr in range(len_trials):
        # No test_data was culled
        # test and train
        Xtest = fliter_p300[:, :, tr, 0:sample_num:5]
        Xtest = np.transpose(Xtest, [1, 0, 2])
        X_test = Xtest.reshape(Xtest.shape[0], chann * sample_num_s)

        choose_tar = [
            i for i in range(each_trial_tar * len_trials)
            if i != (tr * each_trial_tar) and i != (tr * each_trial_tar + 1)
        ]
        choose_nontar = [
            i for i in range((row_plus_col - each_trial_tar) * len_trials)
            if i < (tr * (row_plus_col - each_trial_tar)) or i >=
            (tr * (row_plus_col - each_trial_tar) + 7)
        ]
        target_Temp_test = target_Temp_S2[choose_tar, :]
        nontarget_Temp_test = nontarget_Temp_S2[choose_nontar, :]
        label_tar = np.ones([each_trial_tar * (len_trials - 1)])
        label_nontar = np.zeros([
            (row_plus_col - each_trial_tar) * (len_trials - 1)
        ])
        Temp_test = np.concatenate((target_Temp_test, nontarget_Temp_test),
                                   axis=0)
        label_test = np.concatenate((label_tar, label_nontar), axis=0)
        LDA_model = LDA()
        LDA_model.fit(Temp_test, label_test)
        # predict
        rr = np.dot(X_test, np.transpose(
            LDA_model.coef_)) + LDA_model.intercept_
        # rr = ITCCA(target_Temp, nontarget_Temp, X_test)
        # predict result
        row_max = max(rr[0:row])
        A = [i for i in range(row) if rr[i] == row_max]
        col_max = max(rr[row:col + row])
        B = [i for i in range(row, col + row) if rr[i] == col_max]
        predict_labels[tr, A[0]] = 1
        predict_labels[tr, B[0]] = 1
        predict_lab = A[0] + (B[0] - row) * row
        print('predict_lab = ', predict_lab)
        if (event_labels[tr, :] == predict_labels[tr, :]).all():
            true_num = true_num + 1
        if (event_labels[tr, 0:row] == predict_labels[tr, 0:row]).all():
            true_row_num = true_row_num + 1
        if (event_labels[tr, row:row + col] == predict_labels[tr, row:row +
                                                              col]).all():
            true_col_num = true_col_num + 1
    print('accuracy= ', true_num / len_trials)

    label_tar = np.ones([each_trial_tar * len_trials])
    label_nontar = np.zeros([(row_plus_col - each_trial_tar) * len_trials])
    Temp_S2 = np.concatenate((target_Temp_S2, nontarget_Temp_S2), axis=0)
    label_all = np.concatenate((label_tar, label_nontar), axis=0)

    ### Train model
    LDA_model = LDA()
    LDA_model.fit(Temp_S2, label_all)

    return LDA_model, Temp_S2, label_all


def online_model_predict(data, srate, model, row_plus_col, row, col,
                         stim_interval):

    # data： （68导联+1个标签位）*（时间*采样频率）   data是缓冲区提取出来的数据
    source_data = data
    print(source_data.shape)
    channel_all = 68
    A = [i for i in range(len(source_data[-1, :])) if source_data[-1, i] > 0]
    # print('A=',A)                                    # find labels
    A2 = [i for i in range(1, len(A)) if abs(A[i - 1] - A[i]) > (0.15 * 1000)]
    # print('****',A[A2])
    # B = [i for i in range(len(A2)) if A2[i] > 50]
    # print('B=',B)
    # if len(B) > 2:
    #     trial_latency = A2[B[len(B) - 2] + 1:B[len(B) - 1]]
    # else:                                   # find latency
    trial_latency = [A[A2[i]] for i in range(len(A2)) if A[A2[i]] > 0
                     ]  # event latency + row/col latency + end latency
    print('trial_latency=', trial_latency)
    dataArray = np.zeros([
        row_plus_col, channel_all,
        int((stim_interval[1] - stim_interval[0]) * srate + 1)
    ])
    print(dataArray.shape)
    for i in range(row_plus_col):
        trial_label = int(source_data[-1, trial_latency[i]] - 21)
        print(trial_label)
        ist = int(trial_latency[i] + stim_interval[0] * 1000)
        ie = int(trial_latency[i] + stim_interval[1] * 1000)
        dataArray[trial_label, :, :] = source_data[:channel_all, ist:ie:4]
    # channel selection
    rawdata = np.transpose(
        dataArray,
        [1, 0, 2])  # channels * stim_type(row plus col) * sample_num
    channel_data = chans_pick(rawdata,
                              chans=['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'Oz'])

    # filter
    lpass = 1  # 低通截至频率
    hpass = 12  # 高通截至频率
    filterorder = 3  # 定义带通滤波器的阶数
    fs_p300 = srate  # 生成数字滤波器
    sos = signal.butter(filterorder, [lpass, hpass],
                        btype='band',
                        analog=False,
                        fs=fs_p300,
                        output='sos')
    fliter_p300 = signal.sosfiltfilt(
        sos, channel_data,
        axis=2)  # 6 channels * stim_type(row plus col)  * sample_num

    # for eve in range(1):
    #     event_lab = source_data[channel_all,       # different from offline
    #                             trial_latency[0]]  # event_lab:1-20
    #     [row_id, col_id] = divmod(event_lab, col)
    #     if col_id == 0:
    #         col_id = col
    #         row_id = row_id - 1
    #     event_labels[eve, row_id] = 1
    #     event_labels[eve, row + col_id - 1] = 1

    # predict
    ### LDA predict
    [chann, stim_type, sample_num] = fliter_p300.shape
    predict_labels = np.zeros([1, row_plus_col])
    true_num = 0
    predict_lab = 0
    for tr in range(1):
        # No test_data was culled
        Xtest = fliter_p300[:, :, 0:sample_num:5]
        Xtest = np.transpose(
            Xtest,
            [1, 0, 2])  # stim_type(row plus col)  * 6 channels *  sample_num
        X_test = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
        rr = np.dot(X_test, np.transpose(model.coef_)) + model.intercept_
        # rr = ITCCA(target_Temp, nontarget_Temp, X_test)
        # predict result
        row_max = max(rr[0:row])
        A = [i for i in range(row) if rr[i] == row_max]
        col_max = max(rr[row:col + row])
        B = [i for i in range(row, col + row) if rr[i] == col_max]
        predict_labels[tr, A[0]] = 1
        predict_labels[tr, B[0]] = 1
        predict_lab = A[0] + (B[0] - row) * row
        # if (event_labels[tr, :] == predict_labels[tr, :]).all():
        #     true_num = true_num + 1

    return true_num, predict_lab  # true_num :1(true) or 0 (flase)
    # predict_lab : 1-20


class FeedbackWorker(ProcessWorker):

    def __init__(self, filepath, pick_chs, n_char, row, col, fs_p300,
                 stim_interval, stim_labels, lsl_source_id, timeout,
                 worker_name):
        self.filepath = filepath
        self.pick_chs = pick_chs
        self.n_char = n_char
        self.row = row
        self.col = col
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.fs_p300 = fs_p300
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        raw = P300read_data(filepath=self.filepath)
        print("Loding data successfully")
        self.estimator, Temp_S2, label_all = P300train_predict_model(
            raw,
            n_char=self.n_char,
            row_plus_col=self.row + self.col,
            row=self.row,
            col=self.col,
            fs_p300=self.fs_p300,
            stim_interval=self.stim_interval)

        info = StreamInfo(name='meta_feedback',
                          type='Markers',
                          channel_count=1,
                          nominal_srate=0,
                          channel_format='int32',
                          source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        true_labels, p_labels = online_model_predict(
            data,
            srate=self.fs_p300,
            model=self.estimator,
            row_plus_col=self.row + self.col,
            row=self.row,
            col=self.col,
            stim_interval=self.stim_interval)
        print('true_labels=', true_labels)  # when it's true ,true_labels=1
        p_labels = p_labels + 1
        p_labels = [p_labels]
        print(p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

    def post(self):
        pass


if __name__ == '__main__':
    srate = 1000                                        # 放大器的采样率
    stim_interval = [0.05, 0.80]                        # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_labels = list(range(1, 30))                    # 事件标签
    cnts = 3                                            # .cnt数目
    filepath = "data\\p300"                             # 数据的相对路径
    filepath = os.path.join(os.path.dirname(__file__),filepath)
    pick_chs = ['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'OZ']  # 使用导联
    row_plus_col = 9                                    # 4 * 5
    row = 4
    col = 5
    n_char = 20                                         # char num
    fs_p300 = 250

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(filepath=filepath,
                            pick_chs=pick_chs,
                            n_char=n_char,
                            row=row,
                            col=col,
                            fs_p300=fs_p300,
                            stim_labels=stim_labels,
                            stim_interval=stim_interval,
                            lsl_source_id=lsl_source_id,
                            timeout=5e-2,
                            worker_name=feedback_worker_name)  # 在线处理
    marker = Marker(interval=[0, 6.1], srate=srate, events=[1])     # 打标签全为1

    ns = NeuroScan(device_address=('192.168.1.100', 4000),
                   srate=srate,
                   num_chans=68)  # NeuroScan parameter

    ns.connect_tcp()                                                # 与ns建立tcp连接
    ns.start_acq()                                                  # ns开始采集波形数据

    ns.register_worker(feedback_worker_name, worker,
                       marker)  # register worker来实现在线处理
    ns.up_worker(feedback_worker_name)                              # 开启在线处理进程
    time.sleep(0.5)                                                 # 等待 0.5s

    ns.start_trans()                                                # ns开始截取数据线程，并把数据传递数据给处理进程

    input('press any key to close\n')                               # 任意键关闭处理进程
    ns.down_worker('feedback_worker')                               # 关闭处理进程
    time.sleep(1)                                                   # 等待 1s

    ns.stop_trans()                                                 # ns停止在线截取线程
    ns.stop_acq()                                                   # ns停止采集波形数据
    ns.close_connection()                                           # 与ns断开连接
    ns.clear()
    print('bye')
