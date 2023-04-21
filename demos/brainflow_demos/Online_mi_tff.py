import copy
import socket
import time
from typing import List
import logging
from scipy import signal
import mne
import numpy as np
from metabci.brainflow.amplifiers import Neuracle, TffMarker
from metabci.brainflow.workers import ProcessWorker
from TffModel import TffModel
import pandas as pd
from pathlib import Path


class FeedbackWorker(ProcessWorker):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('192.168.1.100', 20131)
    logging.basicConfig(filename="tff.log", level=logging.DEBUG)

    def __init__(self, timeout, model_list, worker_name, channels: List):
        self.models = model_list
        self.channels = channels
        self.records_list = []
        self.mark_list = [12, 13, 14, 15, 16, 52, 53, 54, 55, 56]
        self.records_list_name = ['原始标签', '预测标签', '置信度', '是否成功切掉']
        self.__time_windows = [(2.0 + index * 0.4) * 1000 for index in range(5)]
        self.__freq_windows = self.__init_freq_window()
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        logging.basicConfig(filename="tff.log", level=logging.DEBUG)
        logging.info(len(self.models))

    def consume(self, data):
        event = data['event']
        data = data['data']
        data: np.ndarray = np.array(data, dtype=np.float64).T
        data = data[:-1]
        data = data.reshape((1, data.shape[0], data.shape[1]))
        index = self.__time_windows.index(data.shape[2])
        pre_label, acc = self.predict(model_index=index, data=data)
        # data = data[self.channels]
        # print(data.shape)
        # p_label, acc = self.predict(model_index=1, data=data)
        # 发送结果给刺激界面
        b_p_label = str(pre_label)
        print("标签为{},置信度为{}".format(pre_label, acc))
        is_cut = 1 if (b_p_label == '1' and (int(event) in self.mark_list[:5])) or (
                b_p_label == '2' and (int(event) in self.mark_list[5:])) else 0
        record_list = [event, b_p_label, acc, is_cut]
        print(record_list)
        self.records_list.append(record_list)
        if b_p_label == '1':
            b_p_label = str(acc).encode()
            self.client_socket.sendto(b_p_label, self.server_address)
        elif b_p_label == '2':
            b_p_label = str(acc + 101).encode()
            self.client_socket.sendto(b_p_label, self.server_address)

    def post(self):
        # 保存数据
        data_frame = pd.DataFrame(data=self.records_list, columns=self.records_list_name)
        save_path = Path('record.csv')
        data_frame.to_csv(path_or_buf=save_path, encoding='utf-8_sig')

    def predict(self, model_index: int, data, sample_rate: int = 1000):

        """
        TODO:可能有各种问题 等待debug
        :param model_index: 一共五个模型 索引
        :param data: 在线接受的数据
        :param sample_rate: 在线的采样率
        :return: final_pre 预测结果 1 or 2 or 3 acc 准确率
         """
        model_list = self.models[model_index]
        mne.filter.resample(x=data, up=200, down=sample_rate)

        vote_list = [0, 0, 0]
        for model in model_list:
            data_copy = copy.deepcopy(data)
            freq_window = self.__freq_windows[model['freq']]
            data_copy = self.bandpass_filter(data_copy, freq_window[0], freq_window[1], sample_rate=200)
            data_transformed = model['csp'].transform(data_copy)
            pre_label = model['svc'].predict(data_transformed)
            pre_label = int(pre_label) - 1
            vote_list[pre_label] += 1
        print(vote_list)
        final_pre = vote_list.index(max(vote_list)) + 1
        acc = round((max(vote_list) / sum(vote_list) * 1.0) * 100)
        return final_pre, acc

    @staticmethod
    def __init_freq_window():
        """
        一共有四十六个频带 后续预计取三十个评分最好的
        :return: 频带窗
        """
        min_freq = 4
        min_freq_max = 10
        max_freq = 36
        min_step = 8
        freq_windows = []
        for low in range(min_freq, min_freq_max + 1, 2):
            for high in range(low + min_step, max_freq + 1, 2):
                freq_windows.append([low, high])
        return freq_windows

    @staticmethod
    def bandpass_filter(eeg_data, freq0, freq1, sample_rate):
        """
        对脑电数据进行带通滤波
        :param eeg_data: 输入的脑电数据，形状为 (channels, times)
        :param freq0: 低频截止频率
        :param freq1: 高频截止频率
        :param sample_rate: 采样率

        :return: eeg_data_filtered (ndarray): 滤波后的脑电数据，形状与原始数据相同
        """
        # 计算滤波器的参数
        wn1 = 2 * freq0 / sample_rate
        wn2 = 2 * freq1 / sample_rate
        b, a = signal.butter(4, [wn1, wn2], 'bandpass')

        # 对脑电数据进行带通滤波
        eeg_data_filtered = signal.filtfilt(b, a, eeg_data, axis=-1)

        return eeg_data_filtered


if __name__ == '__main__':
    # 放大器的采样率
    sample_rate = 1000
    # 导联
    pick_chs = ['C3', 'FC3', 'FCZ', 'FC4', 'CZ',
                'C4', 'CP3', 'CP4']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker_tff'

    tff_model = TffModel(subject_id='12whr', day_num=2)
    models = tff_model.models
    worker = FeedbackWorker(timeout=5e-2, model_list=models, worker_name=feedback_worker_name,
                            channels=pick_chs)  # 在线处理

    marker = TffMarker(sample_rate=sample_rate)
    # worker.pre()

    ns = Neuracle(
        device_address=('127.0.0.1', 8712),
        srate=sample_rate,
        num_chans=9)

    # 与ns建立tcp连接
    ns.connect_tcp()
    # ns开始采集波形数据
    # ns.start_acq()

    # register worker来实现在线处理
    ns.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    ns.up_worker(feedback_worker_name)
    # 等待 1s
    time.sleep(1)

    # ns开始截取数据线程，并把数据传递数据给处理进程
    ns.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    ns.down_worker(feedback_worker_name)
    # 等待 1s
    time.sleep(0.5)

    # ns停止在线截取线程
    ns.stop_trans()

    ns.close_connection()  # 与ns断开连接
    ns.clear()
    print('bye')
