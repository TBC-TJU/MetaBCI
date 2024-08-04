# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
from functools import partial
from metabci.brainflow.amplifiers import Marker, BlueBCI
from metabci.brainflow.workers import ProcessWorker

from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone

from metabci.brainda.datasets import Experiment
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices)
from metabci.brainda.algorithms.decomposition import (
    FBSCCA, generate_filterbank, generate_cca_references)

from scipy.stats import kurtosis

import time
from pylsl import StreamInfo, StreamOutlet

from demos.brainstim_demos.key_mouse_beta import Virtual_Output





class FeedbackWorker(ProcessWorker):
    def __init__(self, lsl_source_id, timeout, worker_name):
        self.lsl_source_id = lsl_source_id

        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        print("-------------------Entering Pre-------------------")
        #创建返回预测目标的lsl stream，在pre中建立，consume中使用
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)

        #创建鼠标/键盘控制器
        self.controller = Virtual_Output()
        self.controller.start()
        self.CMD_list = [ ###这个CMD list需要在范式自定义的时候传入
            'mouse up 200',
            'mouse down 200',
            'mouse left 200',
            'mouse right 200',
            'mouse button_left 1',
            'mouse button_right 1',
        ]
        time.sleep(1)

        print('Waiting connection brainstim...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                # if self.outlet.have_consumers():
                # #不停寻找同lsl_source_id的刺激程序
                break
        print('Connected to brainstim')


        print("start trainning---------------")
        dataset = Experiment()
        delay = 0  # seconds
        self.delay = delay

        channels = ['PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
        srate = 1000  # Hz
        duration = 4  # seconds
        self.duration = duration

        n_bands = 5
        n_harmonics = 4
        events = sorted(list(dataset.events.keys()))
        freqs = [dataset.get_freq(event) for event in events]
        phases = [dataset.get_phase(event) for event in events]

        Yf = generate_cca_references(
            freqs, srate, duration,
            phases=None,
            n_harmonics=n_harmonics)

        start_pnt = dataset.events[events[0]][1][0]
        self.start_pnt = start_pnt

        paradigm = SSVEP(
            srate=srate,
            channels=channels,
            intervals=[(start_pnt + delay, start_pnt + delay + duration + 0.1)],  # more seconds for TDCA
            events=events)

        wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
        ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
        filterbank = generate_filterbank(
            wp, ws, srate, order=4, rp=1)
        filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

        def data_hook(X, y, meta, caches):
            filterbank = generate_filterbank(
                [[8, 90]], [[6, 95]], srate, order=4, rp=1)
            X = sosfiltfilt(filterbank[0], X, axis=-1)
            return X, y, meta, caches

        paradigm.register_data_hook(data_hook)

        set_random_seeds(64)
        l = 5
        models = OrderedDict([
            ('fbscca', FBSCCA(
                filterbank, filterweights=filterweights)),
            # ('fbecca', FBECCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbdsp', FBDSP(
            #     filterbank, filterweights=filterweights)),
            # ('fbtrca', FBTRCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbtdca', FBTDCA(
            #     filterbank, l, n_components=8,
            #     filterweights=filterweights)),
        ])

        X, y, meta = paradigm.get_data(
            dataset,
            subjects=[2],
            return_concat=True,
            n_jobs=1,
            verbose=False)

        set_random_seeds(42)
        loo_indices = generate_loo_indices(meta)

        for model_name in models:
            if model_name == 'fbtdca':
                filterX, filterY = np.copy(X[..., :int(srate * duration) + l]), np.copy(y)
            else:
                filterX, filterY = np.copy(X[..., :int(srate * duration)]), np.copy(y)

            filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

            n_loo = len(loo_indices[2][events[0]])
            loo_accs = []
            # for k in range(n_loo):
            #     train_ind, validate_ind, test_ind = match_loo_indices(
            #         k, meta, loo_indices)
            #     train_ind = np.concatenate([train_ind, validate_ind])
            #
            #     trainX, trainY = filterX[train_ind], filterY[train_ind]
            #     testX, testY = filterX[test_ind], filterY[test_ind]
            #
            #     model = clone(models[model_name]).fit(
            #         trainX, trainY,
            #         Yf=Yf
            #     )
            #     pred_labels = model.predict(testX)
            #     loo_accs.append(
            #         balanced_accuracy_score(testY, pred_labels))
            # print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))

            trainX, trainY = filterX[:], filterY[:]
            self.classfier = clone(models[model_name]).fit(
                    trainX, trainY,
                    Yf=Yf
                )
            print("Ready to start---------------")

    def consume(self, data):
        print("-------------------Entering consume-------------------")
        # if data:
        #     print("data:", data)
        print("inside consume")

        data = np.array(data)

        srate = 1000  # Hz
        interval = [(self.start_pnt + self.delay)*srate, (self.start_pnt + self.delay + self.duration + 0.1)*srate]
        data = np.transpose(data, [1, 0])
        data = data[:-1,:]
        filterbank = generate_filterbank(
            [[8, 90]], [[6, 95]], srate, order=4, rp=1)
        data = sosfiltfilt(filterbank[0], data, axis=-1)
        data = data[np.newaxis, :, int(interval[0]):int(interval[1])]
        data = data[..., :int(srate * self.duration)]

        p_labels, features = self.classfier.predict(data)

        if kurtosis(np.sort(features), axis=-1,fisher=False) > 2:
            p_labels = int(p_labels[0]) + 1

            #实现控制操作
            self.controller.CMD(self.CMD_list[p_labels-1])
            print("CMD:", self.CMD_list[p_labels-1])
            time.sleep(0.5)
            print("moved")

            print('predict_id_paradigm', p_labels)

            #预测标签反馈
            while not self.outlet.have_consumers():
                time.sleep(0.1)
            self.outlet.push_sample([p_labels])
            print("predict label pushed")
        else:
            print("reject")
            # 预测标签反馈
            while not self.outlet.have_consumers():
                time.sleep(0.1)
            self.outlet.push_sample([1])
            print("predict label pushed")


    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 5.14]
    # Label types
    stim_labels = list(range(1, 7))

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels, save_data=False)

    # worker.pre()
    # Set Neuroscan parameters
    bl = BlueBCI(
        device_address=('127.0.0.1', 12345),
        srate=srate,
        num_chans=8, use_trigger=True)


    # Start tcp connection with ns
    bl.connect_tcp()


    # Register worker for online data processing
    bl.register_worker(feedback_worker_name, worker, marker)
    # Start online data processing
    bl.up_worker(feedback_worker_name)
    time.sleep(5) #留时间做pre()

    # Start slicing data and passing data to worker
    bl.start_trans() #根本是调用父类.start()

    input('press any key to close\n')

    #marker.save_as_mat(experiment_name='test2',subject=2)

    bl.down_worker('feedback_worker')
    time.sleep(1)

    # Stop online data retriving of ns
    bl.stop_trans()
    bl.close_connection()
    print('bye')
