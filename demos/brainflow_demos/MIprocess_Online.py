# -*- coding: utf-8 -*-
# License: MIT License
"""
MI Feedback on NeuroScan.

"""
import os
import time
import numpy as np
import mne
import joblib
from mne.io import read_raw_cnt
from mne.filter import resample
from scipy import signal
import socket
import struct
from metabci.brainflow.amplifiers2 import Neuracle, Marker
from metabci.brainflow.ElectroStimulator import ElectroStimulator
import threading

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.utils import upper_ch_names

from sklearn.base import BaseEstimator, ClassifierMixin


# 按照0,1,2,...重新排列标签
def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

def get_chs_id(pick_chs):
    all_chs = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
        'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
        'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2',
        'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2',
        'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'PZ', 
        'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2']
    chs_id = []
    for i in range(len(pick_chs)):
        chs_id.append(all_chs.index(pick_chs[i].upper(),0,len(all_chs)))
    return chs_id

class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y

class SendMessageUdp():

    def __init__(self, server_ip, server_port=9095, client_ip=None, client_port=0):

        self.dest_ip = server_ip
        self.dest_port = server_port
        self.source_ip = socket.gethostbyname(socket.gethostname()) if client_ip is None else client_ip
        self.source_port = client_port

    def start_client(self):
        self.sock_client = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # self.sock_client.bind((self.source_ip, self.source_port))

    def send_message(self, message):
        if isinstance(message, bytes):
            self.sock_client.sendto(message, (self.dest_ip, self.dest_port))
            print("Send:",format(message))
        else:
            try:
                message = struct.pack('B', message)
            except TypeError as err:
                raise TypeError(err.args)
            else:
                self.sock_client.sendto(message, (self.dest_ip, self.dest_port))
                print("Send:", format(message))

    def close_connect(self):
        self.sock_client.close()


# 读取数据
def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        events = mne.events_from_annotations(
            raw, verbose=False)[0]
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
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_picks


# 带通滤波
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new



# 预测标签
def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    X = bandpass(X, 5, 40, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels



class FeedbackWorker(ProcessWorker):
    def __init__(self, pick_chs, stim_interval, stim_labels, srate, lsl_source_id, timeout, worker_name,server_ip,server_port):
        self.ch_ind = get_chs_id(pick_chs)
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.send_result = SendMessageUdp(server_ip,server_port)
        super().__init__(timeout=timeout, name=worker_name)
        self.stimulator = None  # 电刺激器
        self.stim_lock = None  # 线程锁
        self.current_label = None

    def load_model(self):
        # load training model
        self.estimator = joblib.load('D:\\存储\\MetaBCI-master2\\fbcsp44_2.joblib')
        print('**** Model loaded ****')

    # 模型读取
    def pre(self):
        # 模型读取
        self.load_model()
        # 建立处理计算机与刺激计算机之间的数据流
        self.send_result.start_client()
        self.stimulator = ElectroStimulator('COM4')
        self.stim_lock = threading.Lock()  # 在子进程中初始化锁
        print("电刺激器初始化成功")

    def _stimulate(self, channels, params_list, duration=4):
        """电刺激线程函数"""
        with self.stim_lock:
            try:
                # 清除所有已选通道
                for ch in list(self.stimulator._selected_channels):
                    self.stimulator.disable_channel(ch)
                
                # 设置多个通道参数
                for channel, params in zip(channels, params_list):
                    self.stimulator.select_channel(channel)
                    self.stimulator.set_channel_parameters(channel, params)
                self.stimulator.lock_parameters()
                self.stimulator.run_stimulation(duration)
                
            except Exception as e:
                print(f"电刺激控制出错: {e}")

    # 在线处理
    # def consume(self, data):
    def consume(self, payload):
        # 电刺激参数配置
        params_ch1 = {
            ElectroStimulator._Param.current_positive: 10,
            ElectroStimulator._Param.current_negative: 10,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        params_ch2 = {
            ElectroStimulator._Param.current_positive: 13,
            ElectroStimulator._Param.current_negative: 13,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        params_ch3 = {
            ElectroStimulator._Param.current_positive: 11,
            ElectroStimulator._Param.current_negative: 11,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        params_ch4 = {
            ElectroStimulator._Param.current_positive: 12,
            ElectroStimulator._Param.current_negative: 12,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        }
        data, label = payload
        self.current_label = label

        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = int(p_labels)
        p_labels = p_labels + 1
        # p_labels = [p_labels]
        # p_labels = p_labels.tolist()
        print('[{}]'.format(p_labels))
        # 传递在线结果
        self.send_result.send_message(p_labels)
        # 根据标签选择通道
        if self.current_label == 1 and p_labels == 1:
            print("想象正确,激活通道1,2")
            # for ch in range(1,13):  # 0-12通道
            #     self.stimulator.select_channel(ch, enable=False)
            stim_thread = threading.Thread(
                target=self._stimulate,
                args=([1,2], [params_ch1,params_ch2]))   
            stim_thread.start()        
        elif self.current_label == 2 and p_labels == 2:
            print("想象正确,激活通道3,4")
            # for ch in range(1,13):  # 0-12通道
            #     self.stimulator.select_channel(ch, enable=False)
            stim_thread = threading.Thread(
                target=self._stimulate,
                args=([3,4], [params_ch3,params_ch4]))
            stim_thread.start() 
        else:
            print('判断错误')
            return

    def post(self):
        # 关闭电刺激器连接
        if self.stimulator:
            self.stimulator.close()


if __name__ == '__main__':
    # 初始化参数
    srate = 1000  # 放大器的采样率
    stim_interval = [0, 4]  # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_labels = list(range(1, 3))  # 事件标签
    #pick_chs = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4']
    pick_chs = ['FC3', 'FC4', 'C5', 'C4', 'CP3', 'CP4']
    server_ip = '192.168.1.102' # 101为刺激电脑，102为笔记本
    server_port = 9095 # 9095为刺激电脑，8080为笔记本

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # 实例化FeedbackWorker在线流程框架
    worker = FeedbackWorker( pick_chs=pick_chs, stim_interval=stim_interval,
                            stim_labels=stim_labels, srate=srate, lsl_source_id=lsl_source_id,
                            timeout=5e-2, worker_name=feedback_worker_name, server_ip=server_ip,server_port=server_port)  # 在线处理
    # brainflow.amplifiers.Marker
    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels)  # 打标签全为1

    # brainflow.amplifiers.NeuroScan
    nc = Neuracle(
        device_address=('127.0.0.1', 8712),
        srate=srate,
        num_chans=65)  # Neuracle parameter

    nc.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    nc.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)

    nc.start_trans()