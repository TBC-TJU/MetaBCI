# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import pickle
import time
import numpy as np

import mne
import torch
from joblib import parallel
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBTDCA
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin
import serial
import serial.tools.list_ports


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
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_picks


def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]

    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    freqs = np.arange(8, 16, 0.4)
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    model = FBTDCA(filterbank, padding_len=3, n_components=4,
                   filterweights=np.array(filterweights))
    model = model.fit(X, y, Yf)
    return model


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=250, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels, _ = model.predict(X)
    return p_labels

class NeuroScanPort:
    """
    Send tag communication Using parallel port or serial port.

    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.

    Attributes
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.
        port:
            Send tag communication Using parallel port or serial port.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using port to send tags

        from brainstim.utils import NeuroScanPort
        port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
        VSObject.win.callOnFlip(port.setData, 1)
        port.setData(0)

    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        """Send event labels

        Parameters
        ----------
            label:
                The label sent.

        """
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))



# flag = 0
# if flag == 0:
#     print()
#     port_addr = 'COM10'
#     serial_port = NeuroScanPort(port_addr, use_serial=True)
#     flag += 1



class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.trial_count = 0
        self.time_list = []

        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loding train data successfully")
        # 从文件中加载模型对象
        with open("model.pickle", "rb") as file:
            self.estimator = pickle.load(file)
        self.ch_ind = ch_ind

        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        port_addr = 'COM10'
        self.serial_port = NeuroScanPort(port_addr, use_serial=True)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = np.array([int(p_labels + 1)])
        # p_labels = p_labels.tolist()
        p_labels = list(p_labels)
        p = p_labels
        print('predict_id_paradigm', p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

        # ser = serial.Serial(port="COM17", baudrate=115200)
        # # 串口发送 ABCDEFG，并输出发送的字节数。
        # write_len = ser.write("ABCDEFG".encode('utf-8'))

        p = p[0]
        p_labels = hex(p)

        p_labels = p_labels[2:]
        self.serial_port.setData(p_labels)
        print(p_labels)

    def post(self):
        pass

if __name__ == '__main__':
    # Sample rate EEG amplifier

    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0, 0.6]
    # Label types
    stim_labels = list(np.arange(1, 9, 1))
    cnts = 2
    # Data path
    filepath = "data\\ssvep\\sub1"
    runs = list(range(1, cnts+1))
    run_files = ['{:s}\\{:d}.cnt'.format(
        filepath, run) for run in runs]
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ',
                'PO4', 'PO6', 'O1', 'OZ', 'O2']

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name
    )

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)

    ns = NeuroScan(
        device_address=('192.168.1.100', 4000),
        srate=srate,
        num_chans=68)


    # # Start tcp connection with ns
    # ns.connect_tcp()
    # # Start acquire data from ns
    # ns.start_acq()
    # Register worker for online data processing
    # ns.register_worker(feedback_worker_name, worker, marker)
    # Start online data processing
    # ns.up_worker(feedback_worker_name)
    # time.sleep(0.5)
    # Start slicing data and passing data to worker
    # ns.start_trans()
    # input('press any key to close\n')
    # ns.down_worker('feedback_worker')
    # time.sleep(1)
    #
    # # Stop online data retriving of ns
    # ns.stop_trans()
    # ns.stop_acq()
    # ns.close_connection()
    # ns.clear()
    # print('bye')

    ns.command('connect')  # 与ns连接
    ns.command('start_acquire')  # ns开始采集波形数据

    ns.register_worker(feedback_worker_name, worker, marker)  # register worker来实现在线处理
    ns.up_worker(feedback_worker_name)  # 开启处理进程


    ns.command('start_transport')  # ns开始传递数据给处理进程
    time.sleep(5)  # 等待 0.5s
    input('press any key to close\n')  # 任意键关闭处理进程
    ns.down_worker('feedback_worker')  # 关闭处理进程
    time.sleep(1)  # 等待 1s

    ns.command('stop_transport')  # ns停止传递数据给处理进程
    ns.command('stop_acquire')  # ns停止采集波形数据
    ns.command('disconnect')  # 与ns断开连接
    ns.clear()
    print('bye')