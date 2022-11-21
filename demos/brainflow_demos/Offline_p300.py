# -*- coding: utf-8 -*-
# License: MIT License
"""
P300 Feedback on NeuroScan.

"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import mne
from mne import Epochs
from scipy import signal

from datasets import MetaBCIData

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.io import concatenate_raws
import matplotlib.pyplot as plt
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.feature_analysis.time_analysis import TimeAnalysis

from sklearn.base import BaseEstimator, ClassifierMixin

# 选择导联
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

# 读取P300数据
def P300read_data(filepath):
    filelist = []
    for file in os.listdir(filepath):
        filefullpath = os.path.join(filepath, file)
        filelist.append(filefullpath)

    raw_cnts = []
    for file in filelist:
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

# 训练模型和预测标签
def P300train_predict_model(raw, n_char, row_plus_col, row, col, fs_p300,
                            stim_interval):
    # 截取数据
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
    dataArray = np.array(data) 

    rawdata = np.transpose(
        dataArray,
        [2, 0, 1, 3
         ])  # channels * stim_type(row plus col) * trials * sample_num
    channel_data = chans_pick(rawdata,
                              chans=['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'OZ'])

    # 滤波
    lpass = 1               # 低通截至频率
    hpass = 12              # 高通截至频率
    filterorder = 3         # 定义带通滤波器的阶数
    sos = signal.butter(filterorder, [lpass, hpass],
                        btype='band',
                        analog=False,
                        fs=fs_p300,
                        output='sos')
    fliter_p300 = signal.sosfiltfilt(
        sos, channel_data,
        axis=3)  # 6 channels * stim_type(row plus col) * trials * sample_num

    # 寻找目标和非目标模板信号
    each_trial_tar = 2
    len_trials = trial_events.shape[0]
    [chann, stim_type, trials, sample_num] = fliter_p300.shape
    sample_num_s = len(fliter_p300[1, 1, 1, 0:sample_num:5])
    target_Temp = np.zeros([each_trial_tar * len_trials, chann, sample_num_s])
    nontarget_Temp = np.zeros([(row_plus_col - each_trial_tar) * len_trials,
                               chann, sample_num_s])
    target_Temp_1000 = np.zeros([each_trial_tar * len_trials, chann, sample_num])
    nontarget_Temp_1000 = np.zeros([(row_plus_col - each_trial_tar) * len_trials,
                               chann, sample_num])
    event_labels = np.zeros([len_trials, col + row])
    event_labels_ch = np.zeros([len_trials, 1])
    # set event labels
    True_label = np.array([
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 3, 4, 5, 6, 7, 8, 9
    ])
    
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
                target_Temp_1000[tr * each_trial_tar +
                            tar, :, :] = fliter_p300[:, rc, tr, 0:sample_num]
                tar = tar + 1
            else:
                nontarget_Temp[tr * (row_plus_col - each_trial_tar) +
                               nontar, :, :] = fliter_p300[:, rc, tr,
                                                           0:sample_num:5]
                nontarget_Temp_1000[tr * (row_plus_col - each_trial_tar) +
                               nontar, :, :] = fliter_p300[:, rc, tr,
                                                           0:sample_num]
                nontar = nontar + 1

    # change data size : trial *(chan * sample)
    target_Temp_S2 = target_Temp.reshape(each_trial_tar * len_trials,
                                         chann * sample_num_s)
    nontarget_Temp_S2 = nontarget_Temp.reshape(
        (row_plus_col - each_trial_tar) * len_trials, chann * sample_num_s)
    
    # 留一法交叉验证
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
        # predict result
        row_max = max(rr[0:row])
        A = [i for i in range(row) if rr[i] == row_max]
        col_max = max(rr[row:col + row])
        B = [i for i in range(row, col + row) if rr[i] == col_max]
        predict_labels[tr, A[0]] = 1
        predict_labels[tr, B[0]] = 1
        predict_lab = A[0] + (B[0] - row) * row
        if (event_labels[tr, :] == predict_labels[tr, :]).all():
            true_num = true_num + 1
        if (event_labels[tr, 0:row] == predict_labels[tr, 0:row]).all():
            true_row_num = true_row_num + 1
        if (event_labels[tr, row:row + col] == predict_labels[tr, row:row +
                                                              col]).all():
            true_col_num = true_col_num + 1
    acc = true_num / len_trials

    label_tar = np.ones([each_trial_tar * len_trials])
    label_nontar = np.zeros([(row_plus_col - each_trial_tar) * len_trials])
    Temp_S2 = np.concatenate((target_Temp_S2, nontarget_Temp_S2), axis=0)
    label_all = np.concatenate((label_tar, label_nontar), axis=0)

    # Train model
    LDA_model = LDA()
    LDA_model.fit(Temp_S2, label_all)

    return acc, LDA_model, Temp_S2, label_all, nontarget_Temp_1000, target_Temp_1000

# 时域分析
def time_feature(nontarget_Temp, target_Temp):
    # 初始化参数
    srate = 250                                                             # 放大器的采样率
    stim_interval = [(0, 4)]                                                # 截取数据的时间段
    subjects = list(range(1, 2))
    dataset = MetaBCIData(
        subjects=subjects, srate=srate, 
        paradigm='imagery', pattern='imagery')                               # declare the dataset
    paradigm = MotorImagery(
        channels=dataset.channels, 
        events=dataset.events,
        intervals=stim_interval,
        srate=srate
    )                                                                        # declare the paradigm, use recommended Options
    X, y, meta = paradigm.get_data(
        dataset, 
        subjects=subjects,
        return_concat=True, 
        n_jobs=-1, 
        verbose=False)
    
    # brainda.algorithms.feature_analysis.time_analysis.TimeAnalysis
    TimeAna = TimeAnalysis(data=target_Temp, meta=meta, event="left_hand", dataset = dataset, latency = 0)

    fig= plt.figure(1)
    ax = plt.subplot(2,1,1)
    # 画出模板信号及其振幅调用TimeAnalysis.plot_single_trial()
    loc,amp,ax=TimeAna.plot_single_trial(np.mean(target_Temp[:,2,:],axis=0,keepdims=False),sample_num = target_Temp.shape[2],
                                                amp_mark='peak',time_start=0, time_end=target_Temp.shape[2]-1, axes=ax)
    plt.title("Target: Cz",x=0.2, y= 0.86)
    ax = plt.subplot(2,1,2)
    # 画出多试次信号调用TimeAnalysis.plot_multi_trials()
    ax=TimeAna.plot_multi_trials(target_Temp[:,5,:],sample_num = target_Temp.shape[2],axes=ax)
    plt.title("Target: Oz",x=0.2, y= 0.86)

    fig= plt.figure(2)
    ax = plt.subplot(2,1,1)
    # 画出模板信号及其振幅调用TimeAnalysis.plot_single_trial()
    loc,amp,ax=TimeAna.plot_single_trial(np.mean(nontarget_Temp[:,2,:],axis=0,keepdims=False),sample_num = nontarget_Temp.shape[2],
                                            amp_mark='peak',time_start=0, time_end=nontarget_Temp.shape[2]-1, axes=ax)
    plt.title("Nontarget: Cz",x=0.2, y= 0.86)
    ax = plt.subplot(2,1,2)
    # 画出多试次信号调用TimeAnalysis.plot_multi_trials()
    ax=TimeAna.plot_multi_trials(nontarget_Temp[:,5,:],sample_num = nontarget_Temp.shape[2],axes=ax)
    plt.title("Nontarget: Oz",x=0.2, y= 0.86)

    plt.show()

if __name__ == '__main__':
    # 初始化参数
    srate = 1000                                        # 放大器的采样率
    stim_interval = [0.05, 0.80]                        # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_labels = list(range(1, 30))                    # 事件标签
    cnts = 3                                            # .cnt数目
    filepath = "data\\p300\\sub1"                       # 数据的相对路径
    filepath = os.path.join(os.path.dirname(__file__),filepath)
    pick_chs = ['FCZ', 'CZ', 'PZ', 'PO7', 'PO8', 'OZ']  # 使用导联
    row_plus_col = 9                                    # 4 * 5
    row = 4
    col = 5
    n_char = 20                                         # char num
    fs_p300 = 250
    
    # 读取raw
    raw = P300read_data(filepath=filepath)
    print("Loding data successfully")
    
    # 计算离线正确率
    acc, estimator, Temp_S2, label_all , nontarget_Temp, target_Temp= P300train_predict_model(
        raw,
        n_char=n_char,
        row_plus_col=row + col,
        row=row,
        col=col,
        fs_p300=fs_p300,
        stim_interval=stim_interval)
    print("Current Model accuracy:{:.2f}".format(acc))

    # 时域分析
    time_feature(nontarget_Temp, target_Temp)


    