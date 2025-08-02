# -*- coding: utf-8 -*-
# License: MIT License
"""
MI Feedback on Neuracle.

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
    all_chs = ['TP8', 'F4', 'CP2', 'Pz', 'P3', 'AF3', 'TP7', 'C3', 'PO8', 'P4', 'Fp1', 'F5', 'F3', 'F6', 'FC2', 'F1', 'PO5', 'Oz', 'C4', 'HEOR', 'AF7', 'O1', 'Fp2', 'AF8', 'F2', 'CP6', 'T8', 'FC5', 'Fpz', 'CP1', 'F7', 'C5', 'P7', 'PO3', 'FC1', 'PO7', 'FC6', 'FCZ', 'C2', 'Fz', 'O2', 'VEOL', 'FC3', 'FT8', 'C6', 'CP4', 'C1', 'CP5', 'VEOU', 'PO4', 'F8', 'ECG', 'HEOL', 'AF4', 'P6', 'P8', 'P5', 'FT7', 'T7', 'CP3', 'CZ', 'POz', 'PO6', 'FC4']
    chs_id = []
    for i in range(len(pick_chs)):
        chs_id.append(all_chs.index(pick_chs[i].upper(),0,len(all_chs)))
    # print(chs_id)
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
        # raw.resample(250, npad="auto", verbose=False)  # 降采样到250Hz
        # raw.filter(4, 40, l_trans_bandwidth=2, h_trans_bandwidth=5,phase='zero-double')
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
    # ys = label_encoder(ys, labels)

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
    
    # X.resample(250, npad="auto", verbose=False)  # 降采样到250Hz
    # X.filter(4, 40, l_trans_bandwidth=2, h_trans_bandwidth=5,phase='zero-double')
    
    # 降采样
    X = resample(X, up=250, down=srate)
    # # 滤波
    X = bandpass(X, 4, 40, 250)
    # # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    # print(p_labels)
    print(model.predict_proba(X))
    return p_labels

class FeedbackWorker(ProcessWorker):
    def __init__(self, pick_chs, stim_interval, stim_labels, srate, lsl_source_id, timeout, worker_name,server_ip,server_port):
        # self.ch_ind = get_chs_id(pick_chs)
        self.ch_ind = [41,35 ,40, 28 ,18 ,29 ,39 ,33 ,21 ,34 ,30 ,17 ,22 ,
                       16 ,27 ,19 ,24 ,31 ,37 ,26 ,38 ,23 ,32 ,36 ,25, 20]
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.send_result = SendMessageUdp(server_ip,server_port)
        super().__init__(timeout=timeout, name=worker_name)
        self.stimulator = None  # 电刺激器
        self.stim_lock = None  # 线程锁
        self.current_label = None
        # 预定义通道参数
        self.channel_params = {
            1: {
                ElectroStimulator._Param.current_positive: 10,
                ElectroStimulator._Param.current_negative: 10,
                ElectroStimulator._Param.pulse_positive: 250,
                ElectroStimulator._Param.pulse_negative: 250,
                ElectroStimulator._Param.frequency: 50,
                ElectroStimulator._Param.rise_time: 500,
                ElectroStimulator._Param.stable_time: 2000,
                ElectroStimulator._Param.descent_time: 500
            },
            2: {
                ElectroStimulator._Param.current_positive: 10,
                ElectroStimulator._Param.current_negative: 10,
                ElectroStimulator._Param.pulse_positive: 250,
                ElectroStimulator._Param.pulse_negative: 250,
                ElectroStimulator._Param.frequency: 50,
                ElectroStimulator._Param.rise_time: 500,
                ElectroStimulator._Param.stable_time: 2000,
                ElectroStimulator._Param.descent_time: 500
            },
            3: {
                ElectroStimulator._Param.current_positive: 10,
                ElectroStimulator._Param.current_negative: 10,
                ElectroStimulator._Param.pulse_positive: 250,
                ElectroStimulator._Param.pulse_negative: 250,
                ElectroStimulator._Param.frequency: 50,
                ElectroStimulator._Param.rise_time: 500,
                ElectroStimulator._Param.stable_time: 2000,
                ElectroStimulator._Param.descent_time: 500
            },
            4: {
                ElectroStimulator._Param.current_positive: 10,
                ElectroStimulator._Param.current_negative: 10,
                ElectroStimulator._Param.pulse_positive: 250,
                ElectroStimulator._Param.pulse_negative: 250,
                ElectroStimulator._Param.frequency: 50,
                ElectroStimulator._Param.rise_time: 500,
                ElectroStimulator._Param.stable_time: 2000,
                ElectroStimulator._Param.descent_time: 500
            }
        }

    def load_model(self):
        # load training model
        self.estimator = joblib.load('C:\\Users\\86182\\Desktop\\MetaBCI\\fbcsp_0801_1.joblib')
        print('**** Model loaded ****')

    # 模型读取
    def pre(self):
        # 模型读取
        self.load_model()
        # 建立处理计算机与刺激计算机之间的数据流
        self.send_result.start_client()
        self.stimulator = ElectroStimulator('COM5')
        self.stim_lock = threading.Lock()  # 在子进程中初始化锁
        # 预配置所有通道参数
        if self.stimulator:
            try:
                # 配置1-4号通道参数
                for channel in range(1, 5):
                    params = self.channel_params[channel]
                    for param, value in params.items():
                        self.stimulator.set_parameter(channel, param, value)
            except Exception as e:
                print(f"通道参数配置失败: {e}")
        print("电刺激器初始化成功")

    def _stimulate(self, channels, result, duration=3):
        """电刺激线程函数"""
        if self.stimulator is None:
            return 
        with self.stim_lock:
            try:
                # 设置多个通道参数
                for channel in channels:
                    self.stimulator.select_channel(channel, enable=True)
                self.stimulator.lock_parameters()
                self.stimulator.run_stimulation(duration)
            except Exception as e:
                print(f"电刺激控制出错: {e}")
            finally:
                # 电刺激结束后发送结果
                self.send_result.send_message(result)
                for channel in channels:
                    self.stimulator.disable_channel(channel)

    # 在线处理
    # def consume(self, data):
    def consume(self, payload):
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

        # 根据标签选择通道
        if self.current_label == 1 and p_labels == 1:
            print("想象正确,激活通道1,2")
            stim_thread = threading.Thread(
                target=self._stimulate,
                args=([1,2],p_labels))   
            stim_thread.start()        
        elif self.current_label == 2 and p_labels == 2:
            print("想象正确,激活通道3,4")
            stim_thread = threading.Thread(
                target=self._stimulate,
                args=([3,4],p_labels))
            stim_thread.start() 
        else:
            print('判断错误')
            # 判断错误时直接发送结果
            self.send_result.send_message(p_labels)
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
    pick_chs = ['TP8', 'CP2', 'TP7', 'C3', 'FC2', 'C4', 'CP6', 'T8', 'FC5', 'CP1', 'C5', 'FC1', 'FC6', 'FCZ', 'C2', 'FC3', 'FT8', 'C6', 'CP4', 'C1', 'CP5', 'FT7', 'T7', 'CP3', 'CZ', 'FC4']
    server_ip = '192.168.1.102' # 101为刺激电脑，102为笔记本
    server_port = 9095 # 9095为刺激电脑，8080为笔记本

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # 实例化FeedbackWorker在线流程框架
    worker = FeedbackWorker( pick_chs=pick_chs, stim_interval=stim_interval,
                            stim_labels=stim_labels, srate=srate, lsl_source_id=lsl_source_id,
                            timeout=5e-2, worker_name=feedback_worker_name, server_ip=server_ip, server_port=server_port)  # 在线处理
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