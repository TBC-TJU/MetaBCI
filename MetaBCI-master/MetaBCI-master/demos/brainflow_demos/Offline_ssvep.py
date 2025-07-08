# -*- coding: utf-8 -*-
"""
SSVEP offline analysis.
"""
from metabci.brainda.algorithms.decomposition import FBTDCA
from sklearn.base import BaseEstimator, ClassifierMixin
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis \
    import TimeFrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.freq_analysis \
    import FrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.time_analysis \
    import TimeAnalysis
from datasets import MetaBCIData
from mne.filter import resample
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# 对raw操作,例如滤波

def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(7, 55, l_trans_bandwidth=2, h_trans_bandwidth=5,
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

# 训练模型


def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 滤波器组设置
    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]
    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)

    freqs = np.arange(8, 16, 0.4)
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    model = FBTDCA(filterbank, padding_len=3, n_components=4,
                   filterweights=np.array(filterweights))
    model = model.fit(X, y, Yf=Yf)

    return model

# 预测标签


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # FBDSP.predict()预测标签
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

# 时域分析


def time_feature(X, meta, dataset, event, channel, latency=0):
    # brainda.algorithms.feature_analysis.time_analysis.TimeAnalysis
    Feature_R = TimeAnalysis(X, meta, dataset, event=event, latency=latency,
                             channel=channel)

    plt.figure(1)
    # 计算模板信号调用TimeAnalysis.stacking_average()
    data_mean = Feature_R.stacking_average(np.squeeze(
        Feature_R.data[:, Feature_R.chan_ID, :]), _axis=0)
    ax = plt.subplot(2, 1, 1)
    sample_num = int(Feature_R.fs*Feature_R.data_length)
    # 画出模板信号及其振幅调用TimeAnalysis.plot_single_trial()
    loc, amp, ax = Feature_R.plot_single_trial(data_mean,
                                               sample_num=sample_num,
                                               axes=ax,
                                               amp_mark='peak',
                                               time_start=0,
                                               time_end=sample_num-1)
    plt.title("(a)", x=0.03, y=0.86)
    # 画出多试次信号调用TimeAnalysis.plot_multi_trials()
    ax = plt.subplot(2, 1, 2)
    ax = Feature_R.plot_multi_trials(
        np.squeeze(Feature_R.data[:, Feature_R.chan_ID, :]),
        sample_num=sample_num, axes=ax)
    plt.title("(b)", x=0.03, y=0.86)

    # 时域幅值脑地形图
    fig2 = plt.figure(2)
    data_map = Feature_R.stacking_average(Feature_R.data, _axis=0)
    # 调用TimeAnalysis.plot_topomap()
    Feature_R.plot_topomap(data_map, loc, fig=fig2,
                           channels=Feature_R.All_channel, axes=ax)
    plt.show()

# 频域分析


def frequency_feature(X, chan_names, event, SNRchannels, plot_ch, srate=1000):
    # 初始化参数
    channellist = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    chan_nums = []
    for i in range(len(chan_names)):
        chan_nums.append(channellist.index(chan_names[i]))
    X = X[:, chan_nums, :]
    SNRchannels = chan_names.index(SNRchannels)

    # brainda.algorithms.feature_analysis.freq_analysis.FrequencyAnalysis
    Feature_R = FrequencyAnalysis(X, meta, event, srate)

    # 计算模板信号,调用FrequencyAnalysis.stacking_average()
    mean_data = Feature_R.stacking_average(data=[], _axis=0)

    # 计算12Hz刺激下模板信号的功率谱密度
    # 调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[plot_ch])
    plt.plot(f, den)
    plt.text(12, den[f == 12][0], '{:.2f}'.format(
        den[f == 12][0]), fontsize=15)
    plt.text(24, den[f == 24][0], '{:.2f}'.format(
        den[f == 24][0]), fontsize=15)
    plt.text(36, den[f == 36][0], '{:.2f}'.format(
        den[f == 36][0]), fontsize=15)
    plt.title('OZ FFT')
    plt.xlim([0, 60])
    plt.ylim([0, 4])
    plt.xlabel('fre [Hz]')
    plt.ylabel('PSD [V**2]')
    plt.show()


def time_frequency_feature(X, y, chan_names, srate=1000):
    # 初始化参数
    channellist = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                   'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                   'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                   'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                   'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
                   'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
    chan_nums = []
    for i in range(len(chan_names)):
        chan_nums.append(channellist.index(chan_names[i]))
    X = X[:, chan_nums, :]
    index_8hz = np.where(y == 0)
    data_8hz = np.squeeze(X[index_8hz, :, :])
    mean_data_8hz = np.mean(data_8hz, axis=0)
    fs = srate

    # brainda.algorithms.feature_analysis.time_freq_analysis.TimeFrequencyAnalysis
    Feature_R = TimeFrequencyAnalysis(fs)

    # 短时傅里叶变换
    nfft = mean_data_8hz.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx = Feature_R.fun_stft(
        mean_data_8hz, nperseg=256, axis=1, nfft=nfft)
    Zxx_Pz = Zxx[-4, :, :]
    plt.pcolormesh(t, f, np.abs(Zxx_Pz))
    plt.ylim(0, 25)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()

    # 莫雷小波变换
    mean_Pz_data_8hz = mean_data_8hz[-4, :]
    N = mean_Pz_data_8hz.shape[0]
    t_index = np.linspace(0, N / fs, num=N, endpoint=False)
    omega = 2
    sigma = 1
    data_test = np.reshape(mean_Pz_data_8hz, newshape=(
        1, mean_Pz_data_8hz.shape[0]))
    # 调用TimeFrequencyAnalysis.func_morlet_wavelet()
    P, S = Feature_R.func_morlet_wavelet(data_test, f, omega, sigma)
    f_lim = np.array([min(f[np.where(f > 0)]), 30])
    f_idx = np.array(np.where((f <= f_lim[1]) & (f >= f_lim[0])))[0]
    t_lim = np.array([0, 1])
    t_idx = np.array(
        np.where((t_index <= t_lim[1]) & (t_index >= t_lim[0])))[0]
    PP = P[0, f_idx, :]
    plt.pcolor(t_index[t_idx], f[f_idx], PP[:, t_idx])
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.xlim(t_lim)
    plt.ylim(f_lim)
    plt.plot([0, 0], [0, fs / 2], 'w--')
    plt.title(
        ''.join(
            ('Scaleogram (ω = ', str(omega), ' , ', 'σ = ', str(sigma), ')')
            ))
    plt.text(t_lim[1] + 0.04, f_lim[1] / 2, 
             'Power (\muV^2/Hz)', rotation=90,
             verticalalignment='center',
             horizontalalignment='center')
    plt.colorbar()
    plt.show()

    # 希尔伯特变换
    charray = np.mean(data_8hz, axis=1)
    tarray = charray[0, :]
    N1 = tarray.shape[0]
    # 调用TimeFrequencyAnalysis.fun_hilbert()
    analytic_signal, realEnv, imagEnv, angle, envModu = Feature_R.fun_hilbert(
        tarray)

    time = np.linspace(0, N1 / fs, num=N1, endpoint=False)
    plt.plot(time, realEnv, "k", marker='o',
             markerfacecolor='white', label=u"real part")
    plt.plot(time, imagEnv, "b", label=u"image part")
    plt.plot(time, angle, "c", linestyle='-', label=u"angle part")
    plt.plot(time, analytic_signal, "grey", label=u"signal")
    plt.ylabel('Angle or amplitude')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 初始化参数
    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段
    stim_interval = [(0.14, 1.14)]
    subjects = list(range(1, 2))
    paradigm = 'ssvep'

    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    # //.datasets.py中按照metabci.brainda.datasets数据结构自定义数据类MetaBCIData
    # declare the dataset
    dataset = MetaBCIData(
        subjects=subjects, srate=srate,
        paradigm='ssvep', pattern='ssvep')
    paradigm = SSVEP(
        channels=dataset.channels,
        events=dataset.events,
        intervals=stim_interval,
        srate=srate)
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=4,
        verbose=False)
    y = label_encoder(y, np.unique(y))
    print("Loding data successfully")

    # 计算离线正确率
    acc = offline_validation(X, y, srate=srate)     # 计算离线准确率
    print("Current Model accuracy:{:.2f}".format(acc))

    # 时域分析
    time_feature(X[..., :int(srate)], meta, dataset, '11', ['OZ'])  # 1s
    # 频域分析
    frequency_feature(X[..., :int(srate)], pick_chs, '11', 'OZ', -2, srate)
    # 时频域分析
    # time_frequency_feature(X[...,:srate], y,pick_chs)
