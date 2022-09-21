# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np

import mne
from mne.filter import resample

from pylsl import StreamInfo, StreamOutlet

from brainflow.amplifiers import NeuroScan, Marker
from brainflow.workers import ProcessWorker

from brainda.algorithms.decomposition.base import generate_filterbank
from brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from brainda.algorithms.decomposition.trca import EnsembleTRCA
from brainda.algorithms.decomposition.dsp import EnsembleDSP
from brainda.utils import upper_ch_names
from mne.io import read_raw_cnt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline

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

def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        events = mne.events_from_annotations(raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events, event_id=labels, tmin=interval[0], tmax=interval[1], baseline=None, picks=ch_picks, verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    
    return Xs, ys, ch_picks

def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    wp = [
        [6, 88], [14, 88], [22, 88], [20,88], [38, 88]      
    ]
    ws = [
        [4, 90], [12, 90], [20,90], [18, 90], [36,90]
    ]

    filterweights = np.arange(1, 6)**(-1.25) + 0.25    
    filterbank = generate_filterbank(wp, ws, 256)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    model = make_pipeline(*[          # 把模型串联起来
        EnsembleDSP(
            n_components=2, 
            filterbank=filterbank, 
            filterweights=filterweights), 
        # EnsembleTRCA(
        #     n_components=2, 
        #     is_ensemble=True,
        #     filterbank=filterbank, 
        #     filterweights=filterweights), 
        MaxClassifier()])

    model = model.fit(X, y)

    return model

def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels  

def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        
        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels==y_test))
    return np.mean(kfold_accs)

class FeedbackWorker(ProcessWorker): 
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels, srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)
        
    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files, chs=self.pick_chs, 
                                 interval=self.stim_interval, labels=self.stim_labels)
        print("Loding data successfully")
        acc = offline_validation(X, y, srate=self.srate)     # 计算离线准确率
        print("Current Model accuracy:{:.2f}".format(acc))
        self.estimator = train_model(X, y, srate=self.srate)
        self.ch_ind = ch_ind

        info = StreamInfo(
            name='meta_feedback', 
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
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = p_labels + 1
        p_labels = p_labels.tolist()
        print(p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)
    
    def post(self):
        pass

if __name__ == '__main__':
    srate = 1000                                                            # 放大器的采样率
    stim_interval = [0.14, 2.14]                                            # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_labels = list(range(1,21))                                         # 事件标签
    cnts = 2                                                                # .cnt数目
    filepath = "G:\\meta\\已完成\\data\\ssvep"                               # 数据路径
    runs = list(range(1, cnts+1))                                   
    run_files = ['{:s}\\{:d}.cnt'.format(filepath, run) for run in runs]    # 具体数据路径
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']  # 使用导联
    
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'
    
    worker = FeedbackWorker(run_files=run_files, pick_chs=pick_chs, stim_interval=stim_interval, 
                            stim_labels=stim_labels, srate=srate, lsl_source_id=lsl_source_id, 
                            timeout=5e-2, worker_name=feedback_worker_name) # 在线处理
    marker = Marker(interval=stim_interval, srate=srate, events=[1])        # 打标签全为1
    
    ns = NeuroScan(
        device_address=('192.168.1.100', 4000),    
        srate=srate, 
        num_chans=68)                                                       # NeuroScan parameter

    ns.connect_tcp()                                                        # 与ns建立tcp连接
    ns.start_acq()                                                          # ns开始采集波形数据
    
    ns.register_worker(feedback_worker_name, worker, marker)                # register worker来实现在线处理
    ns.up_worker(feedback_worker_name)                                      # 开启在线处理进程
    time.sleep(0.5)                                                         # 等待 0.5s
    
    ns.start_trans()                                                        # ns开始截取数据线程，并把数据传递数据给处理进程
    
    input('press any key to close\n')                                       # 任意键关闭处理进程
    ns.down_worker('feedback_worker')                                       # 关闭处理进程
    time.sleep(1)                                                           # 等待 1s

    ns.stop_trans()                                                         # ns停止在线截取线程
    ns.stop_acq()                                                           # ns停止采集波形数据
    ns.close_connection()                                                   # 与ns断开连接
    ns.clear()
    print('bye')
