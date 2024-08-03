import numpy as np
from PyQt5.QtGui import QIcon
import serial
import threading
import pyttsx3
from scipy.signal import butter, filtfilt
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


    # 静息态基线校正功能
    def baseline_correction(self, data_new, data_quiet):
        """
        Perform baseline correction on EEG data by subtracting the mean of a baseline period.

        Args:
            data_new (ndarray): New EEG data to be corrected.
            data_quiet (ndarray): Baseline period EEG data used for calculating the mean.

        Returns:
            tuple: Two ndarrays containing the baseline-corrected data for both new and quiet periods.
        """
        # Calculate the mean across the time dimension (axis 1)
        quiet_mean = np.mean(data_quiet, axis=1)

        # Use broadcasting to subtract the mean from each trial in both datasets
        data_new_baseline = data_new - quiet_mean
        data_quiet_baseline = data_quiet - quiet_mean

        return data_new_baseline, data_quiet_baseline

    # 滤波功能
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """
        Apply a bandpass filter to 3D data along the last dimension using a Butterworth filter.

        Parameters:
            data (numpy.ndarray): Input data array with shape (trials, channels, time_points).
            lowcut (float): The low cutoff frequency of the filter.
            highcut (float): The high cutoff frequency of the filter.
            fs (float): The sampling frequency of the signal.
            order (int, optional): The order of the filter. Default is 4.

        Returns:
            numpy.ndarray: The filtered data array with the same shape as the input.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        # Check if the input data is 3D
        if data.ndim != 3:
            raise ValueError("Input data must be a three-dimensional array.")

        # Normalize the cutoff frequencies to the Nyquist frequency
        Nyq = fs / 2
        low = lowcut / Nyq
        high = highcut / Nyq

        # Define the Butterworth filter coefficients
        b, a = butter(order, [low, high], btype='bandpass')

        # Initialize an array to hold the filtered data
        filtered_data = np.zeros_like(data)

        # Apply the filter to the last dimension of the data
        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                filtered_data[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])

        return filtered_data

    # 降采样功能
    # (均值，最大，最小)
    def downsample_data(self, data, method='mean', factor=4, fs=1000):
        """
        Downsample the EEG data by applying the specified reduction method across the time dimension.

        Args:
            data (numpy.ndarray): The input EEG data array with shape (trials, channels, time_points).
            method (str): The downsampling method. Options are 'mean', 'max', 'min'. Default is 'mean'.
            factor (int): The downsampling factor. Default is 4.
            fs (int): The original sampling rate of the data in Hz. Default is 1000 Hz.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The downsampled data array.
                - int: The new sampling rate after downsampling.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy ndarray.")

        # Validate that the data has at least 3 dimensions (trials, channels, time_points)
        if data.ndim < 3:
            raise ValueError("Data array must have at least 3 dimensions (trials, channels, time_points).")

        # Check if the time dimension is divisible by the downsampling factor
        if data.shape[2] % factor != 0:
            raise ValueError("The number of time points must be divisible by the downsampling factor.")

        # Reshape the data for downsampling
        reshaped_data = data.reshape(data.shape[0], data.shape[1], -1, factor)

        # Apply the specified downsampling method
        if method == 'mean':
            downsampled_data = reshaped_data.mean(axis=3)
        elif method == 'max':
            downsampled_data = reshaped_data.max(axis=3)
        elif method == 'min':
            downsampled_data = reshaped_data.min(axis=3)
        else:
            raise ValueError("Method must be 'mean', 'max', or 'min'.")

        # Calculate the new sampling rate
        new_fs = fs / factor

        return downsampled_data, new_fs

    # 单个标签数据裁剪函数
    def single_cut_data(self, EEG: dict, mark: int, former_time: float, continue_time: float):
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
        former_sample = int(former_time * EEG['srate'])
        continue_sample = int(continue_time * EEG['srate'])
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
            raw_data[i] = EEG['data'][:, event[0] - former_sample:event[0] + continue_sample]
            raw_label[i] = event[2]

        return raw_data, raw_label

    # 数据裁剪函数
    def cut_data(self, pathname: str, time: list):
        """
        根据指定的时间窗口裁剪数据，并保存为 numpy 文件。
        参数:
            pathname (str): bdf 文件的路径。
            time (list): 包含标签时间窗口的列表，格式为 [[former_time1, continue_time1], [former_time2, continue_time2], ...]。
        返回:
            None
        """
        # 获取bdf文件的数据
        pathname_list = [pathname]
        EEG = readbdfdata(filename=['data.bdf', 'evt.bdf'],
                          pathname=pathname_list)  # 获取地址下的data.bdf和evt.bdf文件，并读取其中的相关数据
        # 计算标签种类
        mark_list = np.unique(EEG['events'][:, 2])
        # 获取数据
        for mark in mark_list:
            # 裁剪数据
            data, label = self.single_cut_data(EEG=EEG, mark=mark, former_time=time[mark - 1][0],
                                               continue_time=time[mark - 1][1])
            # 保存数据为 numpy 文件
            filename = f"{pathname}/data_mark_{mark}.npy"
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










