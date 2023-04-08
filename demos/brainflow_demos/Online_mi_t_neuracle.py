import socket
import time
import numpy as np
import logging
import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import Neuracle, TffMarker, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.csp import FBCSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from scipy import signal


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
                   phase='zero-double')
        events = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events,
                            event_id=labels,
                            tmin=interval[0],
                            tmax=interval[1],
                            baseline=None,
                            picks=ch_picks,
                            verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    return Xs, ys, ch_picks


def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
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
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))

    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 20131)

    def __init__(self,
                 run_files,
                 pick_chs,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.__time_windows = [(2.0 + index * 0.4) * 1000 for index in range(5)]
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)
        logging.basicConfig(filename="tff.log", level=logging.DEBUG)

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loding data successfully")
        acc = offline_validation(X, y, srate=self.srate)  # 计算离线准确率
        print("Current Model accuracy:", acc)
        self.estimator = train_model(X, y, srate=self.srate)
        self.ch_ind = ch_ind

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        data = data[:-1]
        data = data.reshape((1, data.shape[0], data.shape[1]))
        index = self.__time_windows.index(data.shape[2])
        # print(data)
        logging.info(str(time.ctime()) + str(index) + "index")
        print(str(time.ctime()) + str(index) + "index")
        print(data.shape)
        # data = data[self.ch_ind]
        # p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        # p_labels = int(p_labels)
        # p_labels = p_labels + 1
        # # 发送结果给刺激界面
        # b_p_label = str(p_labels)
        # if b_p_label == '0':
        #     b_p_label = '79'.encode()
        # elif b_p_label == '1':
        #     b_p_label = '179'.encode()
        # self.client_socket.sendto(b_p_label, self.server_address)

    def post(self):
        pass


if __name__ == '__main__':
    # 放大器的采样率
    sample_rate = 1000
    stim_interval_offline = [0, 4]
    # 事件标签
    stim_labels_offline = list(range(1, 3))

    cnts = 2  # .cnt数目
    # 数据路径
    BASE_URL = "data\\mi\\sub1"

    runs = list(range(1, cnts + 1))

    run_files = ['{:s}\\{:d}.cnt'.format(
        BASE_URL, run) for run in runs]  # 具体数据路径

    # pick_chs = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ',
    #             'C4', 'CP3', 'CPZ', 'CP4']
    pick_chs = ['C3', 'FC3', 'FCZ', 'FC4', 'CZ',
                'C4', 'CP3', 'CP4']

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(run_files=run_files,
                            pick_chs=pick_chs,
                            stim_interval=stim_interval_offline,
                            stim_labels=stim_labels_offline,
                            srate=sample_rate,
                            lsl_source_id=lsl_source_id,
                            timeout=5e-2,
                            worker_name=feedback_worker_name)  # 在线处理

    marker = TffMarker(sample_rate=1000)
    # marker = Marker(interval=stim_interval_offline, srate=1000,
    #                 events=[52, 53, 54, 55, 56, 12, 13, 14, 15, 16])
    # worker.pre()

    ns = Neuracle(
        device_address=('127.0.0.1', 8712),
        srate=sample_rate,
        num_chans=9)  # NeuroScan parameter

    # 与ns建立tcp连接
    ns.connect_tcp()
    # ns开始采集波形数据
    # ns.start_acq()

    # register worker来实现在线处理
    ns.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    ns.up_worker(feedback_worker_name)
    # 等待 1s
    time.sleep(2)

    # ns开始截取数据线程，并把数据传递数据给处理进程
    ns.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    ns.down_worker('feedback_worker')
    # 等待 1s
    time.sleep(1)

    # ns停止在线截取线程
    ns.stop_trans()

    ns.close_connection()  # 与ns断开连接
    ns.clear()
    print('bye')
