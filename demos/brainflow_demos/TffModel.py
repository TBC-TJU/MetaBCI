import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
import mne
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from typing import List
from sklearn.metrics import accuracy_score
from metabci.brainda.algorithms.decomposition.csp import MultiCSP
import concurrent.futures
import multiprocessing
from scipy import signal
import copy
from mne.decoding.csp import CSP
from metabci.brainflow.logger import get_logger

logger_tff = get_logger("tff_model")


class DataType(Enum):
    BDF = 'bdf'
    CNT = 'cnt'


class TffModel:
    # 路径头
    BASE_URL = '.'

    def __init__(self, subject_id: str, day_num: int, data_type: DataType):
        """
        用对应的被试的离线数据进行建模
        :param subject_id: 被试的编号id 例如 12whr
        :param day_num: 选取被试第几天的数据
        :param data_type: 读取的文件类型
        """

        self.__subject_id = subject_id
        self.__day_num = day_num
        self.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        # 频带窗
        self.__freq_windows = self.__init_freq_window()
        # 一个模型所拥有的频带数
        self.__freq_band_num = 30
        # 原始采样频率
        self.__original_sample_rate = 1000
        # 降采样
        self.__sample_rate = 200
        # 事件id
        self.__event_id = dict(left=1, right=2, normal=3)
        # 导联
        self.__channels = ['C3', 'FC3', 'FCZ', 'FC4', 'CZ',
                           'C4', 'CP3', 'CP4']
        # 时间窗
        self.__time_windows = [[0, 2.0 + index * 0.4] for index in range(5)]
        # raw 列表
        self.__datas = self.__build_datas(data_type)
        # 建模
        self.__models = self.__build_models()

    def predict(self, model_index: int, data, sample_rate: int = 1000):
        """
        TODO:可能有各种问题 等待debug
        :param model_index: 一共五个模型 索引
        :param data: 在线接受的数据
        :param sample_rate: 在线的采样率
        :return: final_pre 预测结果 1 or 2 or 3 acc 准确率
         """
        model_list = self.__models[model_index]
        mne.filter.resample(x=data, up=200, down=sample_rate)
        vote_list = [0, 0, 0]
        for model in model_list:
            data_copy = copy.deepcopy(data)
            freq_window = self.__freq_windows[model['freq']]
            data_copy = self.bandpass_filter(data_copy, freq_window[0], freq_window[1], sample_rate=self.__sample_rate)
            data_transformed = model['csp'].transform(data_copy)
            pre_label = model['svc'].predict(data_transformed)
            pre_label = int(pre_label) - 1
            vote_list[pre_label] += 1
        final_pre = vote_list.index(max(vote_list)) + 1
        acc = max(vote_list) / sum(vote_list)
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

    # 对五个时间窗建立五个模型
    def __build_models(self):
        # 存储五个时间窗对应的model
        model_time_list = []
        for time_window in self.__time_windows:
            # 存储四个文件读出来的epochs
            epochs_list = []
            # 存储一个时间窗对应的30个csp和svc
            csp_svc_freq_list = []
            for raw in self.__datas:
                events, event_id = mne.events_from_annotations(raw)
                # 因为打标签为3的那一组数据的event_id为1 所以做一下特殊处理
                if list(event_id.keys())[0] == '3':
                    events[:, 2] = 3
                    event_id = {'3': 3}
                epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=time_window[0],
                                    tmax=time_window[1] - 1.0 / raw.info['sfreq'], baseline=None, preload=True)
                epochs_list.append(epochs)
            merged_epochs: mne.EpochsArray = mne.concatenate_epochs(epochs_list)
            # 存储每个频带的索引以及准确率
            acc_freq_list = []
            # 存储任务对象
            tasks = []
            # 多进程处理交叉验证
            with multiprocessing.Pool(processes=10) as pool:
                for freq_window in self.__freq_windows:
                    index = self.__freq_windows.index(freq_window)
                    task = pool.apply_async(
                        TffModel.process_task, args=(index, merged_epochs, freq_window))
                    tasks.append(task)

                for task in tasks:
                    result = task.get()
                    acc_freq_list.append(result)
            # 排序 建模
            acc_freq_list.sort(key=lambda acc_freq: acc_freq['acc'], reverse=True)
            # 打日志
            logger_tff.info('{} , {}'.format(time.ctime(), acc_freq_list))
            # 对前三十频带个进行建模
            for index in range(0, self.__freq_band_num):
                freq_window = self.__freq_windows[acc_freq_list[index]['index']]
                m_merged_epochs: mne.EpochsArray = merged_epochs.copy()
                m_merged_epochs.filter(l_freq=freq_window[0], h_freq=freq_window[1])
                train_x, train_y = TffModel.read_epochs(merged_epochs)
                csp = CSP(n_components=4, reg=None, log=None, norm_trace=False)
                # csp = MultiCSP(n_components=4, multiclass='ovr')
                csp.fit(train_x, train_y)
                train_x_csp = csp.transform(train_x)
                svc = SVC(kernel='linear', C=1, decision_function_shape='ovo')
                # svc = SVC(kernel='linear', C=1, decision_function_shape='ovr')
                svc.fit(train_x_csp, train_y)
                csp_svc_freq = {'csp': csp, 'svc': svc, 'freq': acc_freq_list[index]['index']}
                csp_svc_freq_list.append(csp_svc_freq)
            model_time_list.append(csp_svc_freq_list)
        return model_time_list

    @property
    def models(self):
        return self.__models

    @property
    def freq_window(self):
        return self.__freq_windows

    def __build_datas(self, data_type: DataType
                      ):
        datas = []
        if data_type == DataType.CNT:
            raw_paths = [
                Path("{:s}/{:s}/day{:d}/graz1/{:d}.cnt".format(self.BASE_URL, self.__subject_id, self.__day_num, index))
                for
                index in
                range(1, 5)]
            for raw_path in raw_paths:
                raw: mne.io.Raw = mne.io.read_raw_cnt(str(raw_path), preload=True)
                # 降采样
                raw.resample(self.__sample_rate)
                # 选取导联
                raw = raw.pick_channels(self.__channels)
                datas.append(raw)
            logger_tff.info('loaded the data')
        elif data_type == DataType.BDF:
            raw_paths = [
                dict(data_path=Path(
                    "{:s}/{:s}/day{:d}/graz1/{:d}.bdf".format(self.BASE_URL, self.__subject_id, self.__day_num, index)),
                    event_path=Path(
                        "{:s}/{:s}/day{:d}/graz1/evt{:d}.bdf".format(self.BASE_URL, self.__subject_id, self.__day_num,
                                                                     index)))
                for index in range(1, 5)]
            for raw_path in raw_paths:
                raw: mne.io.Raw = mne.io.read_raw_cnt(str(raw_path[0]), preload=True)
                raw_event: mne.io.Raw = mne.io.read_raw_bdf(str(raw_path[1]), preload=True)
                raw.set_annotations(raw_event.annotations)
                # 降采样
                raw.resample(self.__sample_rate)
                # 选取导联
                raw = raw.pick_channels(self.__channels)
                datas.append(raw)
            return datas

    @staticmethod
    def read_epochs(epochs_array: mne.EpochsArray):
        train_x = []
        train_y = []
        for index in range(0, len(epochs_array)):
            data = epochs_array[index].get_data()[0]
            train_x.append(data)
            label = epochs_array[index].events[0][-1]
            train_y.append(label)
        return np.array(train_x, dtype=np.float64), np.array(train_y, dtype=np.float64)

    @staticmethod
    def process_task(index: int, merged_epochs: mne.EpochsArray, freq_window: List[int]):
        m_merged_epochs: mne.EpochsArray = merged_epochs.copy()
        m_merged_epochs.filter(l_freq=freq_window[0], h_freq=freq_window[1])
        # 90个epoch 分成十组 进行交叉验证
        k = 10
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        acc = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(m_merged_epochs)):
            # train_idx 里面有81个索引 test_idx 里面有9个索引 循环十次
            train_data, test_data = m_merged_epochs[train_idx], m_merged_epochs[test_idx]
            # TODO:训练模型
            train_x, train_y = TffModel.read_epochs(train_data)
            test_x, test_y = TffModel.read_epochs(test_data)
            csp = CSP(n_components=4, reg=None, log=None, norm_trace=False)
            # csp = MultiCSP(n_components=4, multiclass='ovr')
            csp.fit(train_x, train_y)

            train_x_csp = csp.transform(train_x)
            test_x_csp = csp.transform(test_x)

            svc = SVC(kernel='linear', C=1, decision_function_shape='ovo')
            # svc = SVC(kernel='linear', C=1, decision_function_shape='ovr')
            svc.fit(train_x_csp, train_y)
            train_y_pre = svc.predict(test_x_csp)
            # TODO:评估模型
            acc += accuracy_score(test_y, train_y_pre)
        acc_mean = (round((acc / k) * 100)) / 100
        result = {'index': index, 'acc': acc_mean}
        return result

    @staticmethod
    def bandpass_filter(eeg_data, freq0, freq1, sample_rate):
        """
        对脑电数据进行带通滤波
        :param eeg_data: 输入的脑电数据，形状为 (channels, times)
        :param freq0: 低频截止频率
        :param freq1: 高频截止频率
        :param sample_rate: 采样率
        :return eeg_data_filtered (ndarray): 滤波后的脑电数据，形状与原始数据相同
        """
        # 计算滤波器的参数
        wn1 = 2 * freq0 / sample_rate
        wn2 = 2 * freq1 / sample_rate
        b, a = signal.butter(4, [wn1, wn2], 'bandpass')

        # 对脑电数据进行带通滤波
        eeg_data_filtered = signal.filtfilt(b, a, eeg_data, axis=-1)

        return eeg_data_filtered


if __name__ == '__main__':
    tff_model = TffModel(subject_id='12whr', day_num=2, data_type=DataType.CNT)
