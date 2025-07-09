# -*- coding: utf-8 -*-
# License: MIT License
"""
MI offline analysis.

"""
from metabci.brainda.algorithms.decomposition.csp import FBCSP
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis \
    import TimeFrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.freq_analysis \
    import FrequencyAnalysis
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from datasets import MetaBCIData
from mne.filter import resample
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# 对raw操作,例如滤波

def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

# 按照0,1,2,...重新排列标签


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y

# 带通滤波


def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

# 训练模型


def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    # X = bandpass(X, 6, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # brainda.algorithms.decomposition.csp.MultiCSP
    wp = [(4, 8), (8, 12), (12, 30)]
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=256, order=4, rp=0.5)
    # model = make_pipeline(
    #     MultiCSP(n_components = 2),
    #     LinearDiscriminantAnalysis())
    model = make_pipeline(*[
        FBCSP(n_components=5,
              n_mutualinfo_components=4,
              filterbank=filterbank),
        SVC()
    ])
    # fit()训练模型
    model = model.fit(X, y)

    return model

# 预测标签


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    X = bandpass(X, 8, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels

# 计算离线正确率


def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))

    kfold_accs = []
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)       # 留一法交叉验证
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)          # 训练模型
        p_labels = model_predict(X_test, srate=srate, model=model)  # 预测标签
        kfold_accs.append(np.mean(p_labels == y_test))                # 记录正确率

    return np.mean(kfold_accs)

# 频域分析


def frequency_feature(X, meta, event, srate=1000):
    # brainda.algorithms.feature_analysis.freq_analysis.FrequencyAnalysis
    Feature_R = FrequencyAnalysis(X, meta, event=event, srate=srate)

    # 计算模板信号,调用FrequencyAnalysis.stacking_average()
    mean_data = Feature_R.stacking_average(data=[], _axis=0)

    plt.subplot(121)
    # 计算PSD值,调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[8])  # C3
    plt.plot(f, den*5)
    plt.xlim(0, 35)
    plt.ylim(0, 0.3)
    plt.title("right_hand :C3")
    plt.ylabel('PSD [V**2]')
    plt.xlabel('Frequency [Hz]')
    # plt.show()

    plt.subplot(122)
    # 计算PSD值,调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[12])  # C4
    plt.plot(f, den*5)
    plt.xlim(0, 35)
    plt.ylim(0, 0.3)
    plt.title("right_hand :C4")
    plt.ylabel('PSD [V**2]')
    plt.xlabel('Frequency [Hz]')
    plt.show()

# 时频分析


def time_frequency_feature(X, y, srate=1000):
    # brainda.algorithms.feature_analysis.time_freq_analysis.TimeFrequencyAnalysis
    TimeFreq_Process = TimeFrequencyAnalysis(srate)

    # 短时傅里叶变换  左手握拳
    index_8hz = np.where(y == 0)                          # y=0 左手
    data_8hz = np.squeeze(X[index_8hz, :, :])
    mean_data_8hz = np.mean(data_8hz, axis=0)
    nfft = mean_data_8hz.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx1 = TimeFreq_Process.fun_stft(
        mean_data_8hz, nperseg=1000, axis=1, nfft=nfft)
    Zxx_Pz1 = Zxx1[8, :, :]                             # 导联选择4：C3
    Zxx_Pz2 = Zxx1[12, :, :]                             # 导联选择6：C4
    # 时频图
    plt.subplot(321)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz1))
    plt.ylim(0, 45)
    plt.title('STFT Left C3')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    plt.subplot(322)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz2))
    plt.ylim(0, 45)
    plt.title('STFT Left C4')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    # 短时傅里叶变换  右手握拳
    index_8hz2 = np.where(y == 1)                         # y=0 右手
    data_8hz2 = np.squeeze(X[index_8hz2, :, :])
    mean_data_8hz2 = np.mean(data_8hz2, axis=0)
    nfft = mean_data_8hz2.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx2 = TimeFreq_Process.fun_stft(
        mean_data_8hz2, nperseg=1000, axis=1, nfft=nfft)
    Zxx_Pz3 = Zxx2[8, :, :]                             # 导联选择3：C3
    Zxx_Pz4 = Zxx2[12, :, :]                             # 导联选择5：C4
    # 时频图
    plt.subplot(323)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz3))
    plt.ylim(0, 45)
    plt.title('STFT Right C3')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    plt.subplot(324)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz4))
    plt.ylim(0, 45)
    plt.title('STFT Right C4')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    plt.show()

    # 脑地形图
    temp_map = np.mean(Zxx1, axis=1)
    temp = np.mean(temp_map, axis=1)
    # 调用TimeFrequencyAnalysis.fun_topoplot()
    TimeFreq_Process.fun_topoplot(np.abs(temp), pick_chs)

    # topomap
    temp_map = np.mean(Zxx2, axis=1)
    temp = np.mean(temp_map, axis=1)
    # 调用TimeFrequencyAnalysis.fun_topoplot()
    TimeFreq_Process.fun_topoplot(np.abs(temp), pick_chs)

    # topomap
    # temp_map = np.mean(Zxx3, axis=1)
    # temp = np.mean(temp_map, axis=1)
    # # 调用TimeFrequencyAnalysis.fun_topoplot()
    # TimeFreq_Process.fun_topoplot(np.abs(temp), pick_chs)


if __name__ == '__main__':
    # 初始化参数
    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段
    stim_interval = [(0, 4)]
    subjects = list(range(1, 2))
    paradigm = 'imagery'
    event = 'right_hand'
    pick_chs = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2',
                'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5',
                'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']

    # //.datasets.py中按照metabci.brainda.datasets数据结构自定义数据类MetaBCIData
    dataset = MetaBCIData(
        subjects=subjects, srate=srate,
        paradigm='imagery', pattern='imagery')  # declare the dataset
    paradigm = MotorImagery(
        channels=dataset.channels,
        events=dataset.events,
        intervals=stim_interval,
        srate=srate)
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=2,
        verbose=False)
    y = label_encoder(y, np.unique(y))
    print("Loding data successfully")

    # 计算离线正确率
    acc = offline_validation(X, y, srate=srate)     # 计算离线准确率
    print("Current Model accuracy:", acc)

    # 频域分析
    frequency_feature(X, meta, event, srate)
    # 时频域分析
    time_frequency_feature(X, y, srate)
