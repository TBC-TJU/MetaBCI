import os

import numpy as np
from PyQt5.QtGui import QIcon
import serial
import threading
import pyttsx3
from scipy.signal import butter, filtfilt, detrend
from .readbdfdata import readbdfdata

# --------------------------------------界面参数------------------------------------
# 设置参数
ICO_path = './Images/brain128.ico'
# 窗口图标更新
def Form_QIcon(formclass):
    formclass.setWindowIcon(QIcon(ICO_path))

# ---------------------------------------打标功能模块------------------------------------------
# 范式标签输出
def send_command(command: list, com: str, enable:bool):
    try:
        if enable:
            sObject = serial.Serial(com, 115200)
            sObject.write(bytes(command))
            sObject.close()
            print('输出：' + str(command))
        else:
            print('当前无打标输出')
    except:
        print('没找到' + com + '端口')

# ------------------------------------------语音模块------------------------------------------
# 语音播报类搭建
class Voice_thread(threading.Thread):
    def __init__(self, sound_text):
        super().__init__()
        self.sound_text = sound_text
        self.daemon = True

    def run(self):
        engine = pyttsx3.init()
        engine.say(self.sound_text)
        engine.runAndWait()
        engine.stop()

# ----------------------------------------预处理功能模块------------------------------------------
class Preprocess_function:
    def __init__(self):
        ...


    # 基线校正和去趋势化功能
    def baseline_correction(self, eeg_signals):
        # 均值基线校正
        mean_baseline = eeg_signals.mean(axis=1, keepdims=True)  # 计算每个通道的均值，keepdims=True 保持维度
        eeg_signals_baseline_corrected = eeg_signals - mean_baseline
        '''
        # # 去趋势化
        # eeg_signals_detrend = np.empty_like(eeg_signals_baseline_corrected)
        # for ch in range(eeg_signals_baseline_corrected.shape[0]):
        #     eeg_signals_detrend[ch, :] = detrend(eeg_signals_baseline_corrected[ch, :], type='linear')
        # eeg_signals_detrend 现在包含了基线校正和去趋势化后的信号
        # return eeg_signals_detrend
        '''
        return eeg_signals_baseline_corrected

    # 滤波功能
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        # Normalize the cutoff frequencies to the Nyquist frequency
        Nyq = fs / 2
        low = lowcut / Nyq
        high = highcut / Nyq

        # Define the Butterworth filter coefficients
        b, a = butter(order, [low, high], btype='bandpass')

        # Initialize an array to hold the filtered data
        filtered_data = np.zeros_like(data)

        # Apply the filter to the last dimension of the data
        for channel in range(data.shape[0]):
            filtered_data[channel, :] = filtfilt(b, a, data[channel, :])

        return filtered_data

    # 降采样功能
    # (均值，最大，最小)
    def downsample_data(self, data, method='mean', factor=4, fs=1000):
        # Reshape the data for downsampling
        reshaped_data = data.reshape(data.shape[0], -1, factor)

        # Apply the specified downsampling method
        if method == 'mean':
            downsampled_data = reshaped_data.mean(axis=2)
        elif method == 'max':
            downsampled_data = reshaped_data.max(axis=2)
        elif method == 'min':
            downsampled_data = reshaped_data.min(axis=2)
        else:
            raise ValueError("Method must be 'mean', 'max', or 'min'.")

        # Calculate the new sampling rate
        new_fs = fs / factor

        return downsampled_data, new_fs

    # 单个标签数据裁剪函数
    def single_cut_data(self, EEG: dict, mark: int, former_time: float, continue_time: float, fs: int, data, factor: int):
        """
        从原始 EEG 数据中裁剪出指定标签的单个实验数据。

        参数:
            EEG (dict): 包含原始 EEG 数据的字典，应至少包含 'data', 'events', 'srate', 'nchan' 键。
            mark (int): 指定的标签。
            former_time (float): 裁剪窗口的前置时间（秒）。
            continue_time (float): 裁剪窗口的持续时间（秒）。

        返回:
            tuple: 包含裁剪后的原始数据和标签的元组，格式为 (raw_data, raw_label)。
                raw_data (ndarray): 裁剪后的原始数据，形状为 (trial, channel, sample)。
                raw_label (ndarray): 裁剪后的原始数据标签，形状为 (trial,)。
        """
        # 计算裁剪窗口的样本数
        former_sample = int(former_time * fs)
        continue_sample = int(continue_time * fs)
        sample = former_sample + continue_sample
        # 获取指定标签的实验次数
        trials = EEG['events'][EEG['events'][:, 2] == mark]
        num_trials = len(trials)
        # 获取数据通道数
        num_channels = EEG['nchan']
        # 初始化裁剪后的原始数据和标签
        raw_data = np.zeros((num_trials, num_channels, sample))
        raw_label = np.zeros(num_trials)
        # 循环提取数据
        for i, event in enumerate(trials):
            raw_data[i] = data[:, int(event[0]/factor) - former_sample:int(event[0]/factor) + continue_sample]
            raw_label[i] = event[2]

        return raw_data, raw_label

    # 数据裁剪函数
    def cut_data(self, pathname: str, time: list, mark_list: list, fs: int, loaddata, factor: int):
        """
        根据指定的时间窗口裁剪数据，并保存为 numpy 文件。
        参数:
            pathname (str): bdf 文件的路径。
            time (list): 包含标签时间窗口的列表，格式为 [[former_time1, continue_time1], [former_time2, continue_time2], ...]。
            mark_list(list):包含标签的列表，格式为[label1, label2,...]
        返回:
            None
        """
        # 获取bdf文件的数据
        pathname_list = [pathname]
        EEG = readbdfdata(filename=['data.bdf', 'evt.bdf'],
                          pathname=pathname_list)  # 获取地址下的data.bdf和evt.bdf文件，并读取其中的相关数据
        # 计算标签种类
        # mark_list = np.unique(EEG['events'][:, 2])
        # 获取数据
        for index, mark in enumerate(mark_list):
            # 裁剪数据
            data, label = self.single_cut_data(EEG=EEG, mark=mark, former_time=time[index][0],
                                               continue_time=time[index][1], fs=fs, data=loaddata, factor=factor)
            # 保存数据为 numpy 文件
            # 构建新文件夹的完整路径
            new_folder_path = os.path.join(pathname, "new_data")
            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            # 构造.npy文件的完整保存路径
            filename = os.path.join(new_folder_path, f"data_mark_{mark}.npy")
            np.save(filename, data)

    # 数据事件检测
    def data_check(self, pathname: str):
        pathname_list = [pathname]
        EEG = readbdfdata(filename=['data.bdf', 'evt.bdf'],
                          pathname=pathname_list)  # 获取地址下的data.bdf和evt.bdf文件，并读取其中的相关数据
        dataset: dict = {
            'data_shape': str(EEG['data'].shape),
            'channel_names': str(EEG['ch_names']),
            'label_list': str(np.unique(EEG['events'][:, 2])),
            'fs': str(EEG['srate']),
            'channel_number': str(EEG['nchan'])
        }
        return dataset

    def data_load(self, pathname: str):
        pathname_list = [pathname]
        EEG = readbdfdata(filename=['data.bdf', 'evt.bdf'],
                          pathname=pathname_list)  # 获取地址下的data.bdf和evt.bdf文件，并读取其中的相关数据

        return EEG['data'], EEG['srate'], EEG['ch_names']











