# -*- coding: utf-8 -*-
"""
SSVEP Feedback on HTOnlineSystem.

"""
import time
import numpy as np

import mne
from mne.filter import resample

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank,
    generate_cca_references,
)
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.cca import FBTRCA
from metabci.brainflow.amplifiers import HTOnlineSystem, Marker
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = y == label
        new_y[ix] = i
    return new_y


def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        events = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False
        )[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=labels,
            tmin=interval[0],
            tmax=interval[1],
            baseline=None,
            picks=ch_picks,
            verbose=False,
        )

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_picks


def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]

    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    freqs = np.arange(8, 16, 0.4)
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    model = FBTRCA(
        filterbank, n_components=4, ensemble=True, filterweights=np.array(filterweights)
    )
    model = model.fit(X, y, Yf)

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
        kfold_accs.append(np.mean(p_labels == y_test))
    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    def __init__(
        self,
        run_files,
        pick_chs,
        stim_interval,
        stim_labels,
        srate,
        lsl_source_id,
        timeout,
        worker_name,
    ):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        # 训练离线模型
        print("Train model process complete.")

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        print(data.shape)
        data = data[self.pick_chs]
        print(data.shape)

    def post(self):
        pass


if __name__ == "__main__":
    srate = 1000  # Sample rate EEG amplifier
    stim_interval = [0.0, 0.5]
    stim_labels = [1]  # Label types

    # Data path
    cnts = 1
    filepath = "data\\train\\sub1"
    runs = list(range(1, cnts + 1))
    run_files = ["{:s}\\{:d}.cnt".format(filepath, run) for run in runs]

    # pick_chs也可以直接使用导联索引
    pick_chs = ["CPz", "PZ", "O1", "OZ", "O2"]

    ####################### 在线设置#############################
    # Set HTOnlineSystem parameters
    ht = HTOnlineSystem(
        device_address=("192.168.1.110", 7110),
        srate=srate,
        packet_samples=100,
        num_chans=32,
    )

    ht.connect_tcp()  # Start tcp connection with ht
 
    # 如果pick_chs是导联列表，则必须先建立tcp连接再初始化worker
    # 如果pick_chs是导联索引列表，则不需要以下查找索引代码，而且初始化worker可以在tcp连接前完成
    all_name_chs = ht.get_name_chans()
    index = []
    for ch in pick_chs:
        try:
            index.append(all_name_chs.index(ch))
        except ValueError:
            print("Channel not found in the setting.")

    lsl_source_id = "meta_online_worker"
    feedback_worker_name = "feedback_worker"

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=index,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name,
    )
    marker = Marker(interval=stim_interval, srate=srate, events=[1])

    ht.start_acq()  # Start acquire data from ht
    ht.register_worker(feedback_worker_name, worker, marker)
    ht.up_worker(feedback_worker_name)  # Start online data processing
    time.sleep(0.5)

    input("press any key to close\n")
    ht.down_worker(feedback_worker_name)
    time.sleep(1)

    # Stop online data retriving of ht
    ht.stop_acq()
    ht.close_connection()
    ht.clear()
    print("bye")
