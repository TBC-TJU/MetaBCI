import os
import numpy as np
from PyQt5.QtGui import QIcon
import serial
import threading
import pyttsx3
from scipy.signal import butter, filtfilt, welch
from .neuracle_lib.readbdfdata import readbdfdata

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


# ---------------------------------------在线数据处理功能模块-----------------------------------------
# 在线数据处理类
class Process:
    def __init__(self):
        self.frequent_arrange = [
            [0.5, 4],
            [4, 7],
            [8, 12],
            [13, 30]
        ]
        # 定义采集原始数据对应位置的导联
        self.weight = {'Fpz': 1, 'Fp1': 1, 'Fp2': 1, 'AF3': 1, 'AF4': 1, 'AF7': 1, 'AF8': 1, 'Fz': 1,
                       'F1': 1, 'F2': 1, 'F3': 1, 'F4': 1, 'F5': 1, 'F6': 1, 'F7': 1, 'F8': 1,
                       'FCz': 1, 'FC1': 1, 'FC2': 1, 'FC3': 1, 'FC4': 1, 'FC5': 1, 'FC6': 1, 'FT7': 1,
                       'FT8': 1, 'Cz': 1, 'C1': 1, 'C2': 1, 'C3': 1, 'C4': 1, 'C5': 1, 'C6': 1,
                       'T7': 1, 'T8': 1, 'CP1': 1, 'CP2': 1, 'CP3': 1, 'CP4': 1, 'CP5': 1, 'CP6': 1,
                       'TP7': 1, 'TP8': 1, 'Pz': 1, 'P3': 1, 'P4': 1, 'P5': 1, 'P6': 1, 'P7': 1,
                       'P8': 1, 'POz': 1, 'PO3': 1, 'PO4': 1, 'PO5': 1, 'PO6': 1, 'PO7': 1, 'PO8': 1,
                       'Oz': 1, 'O1': 1, 'O2': 1
                  }
        # 定义地形图每个导联的位置坐标，此处为全脑地形图的导联坐标
        self.ch_pos = [
                                            [-20, 76], [0, 80], [20, 76],
                                      [-38, 67], [-24, 62], [24, 62], [38, 67],
            [-57, 52], [-42, 49], [-27, 46], [-12, 44], [0, 43], [12, 44], [27, 46], [42, 49], [57, 52],
            [-70, 37], [-53, 33], [-32, 30], [-15, 28], [0, 27], [15, 28], [32, 30], [53, 33], [70, 37],
            [-80, 10], [-57, 10], [-36, 10], [-16, 10], [0, 10], [16, 10], [36, 10], [57, 10], [80, 10],
              [-75, -17], [-54, -13], [-35, -10], [-18, -8], [18, -8], [35, -10], [54, -13], [75, -17],
                    [-66, -35], [-49, -32], [-25, -29], [0, -30], [25, -29], [49, -32], [66, -35],
                    [-49, -56], [-33, -48], [-20, -51], [0, -53], [20, -51], [33, -48], [49, -56],
                                          [-28, -70], [0, -73], [28, -70]
        ]
        # 定义每个导联的名称,此处为全脑导联的名称
        self.EEG_name_list = [
                                    'Fp1', 'Fpz', 'Fp2',
                                 'AF7', 'AF3', 'AF4', 'AF8',
                     'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                      'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                     'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8',
                         'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
                      'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
                                       'O1', 'Oz', 'O2'
            ]
        # 定义C3-C4局部模式的保留导联名称
        self.C3_C4_name_list = [
                'C3', 'C1', 'Cz', 'C2', 'C4',
                'CP3', 'CP1', 'CP2', 'CP4',
            ]
    def calculate_avg_power(self, data, low: int, high: int, fs: int = 250, nperseg: int = 128):
        """
        计算数据的指定频率范围内的平均功率谱密度。
        参数：
            data (list of numpy.ndarray): 输入数据数组,此数组为二维数组，一般为[channel, sample]。
            low (float): 指定频率范围的低频率。
            high (float): 指定频率范围的高频率。
            fs (float): 采样频率，默认为1000 Hz。
            nperseg (int): 每个段的长度，默认为1024。
        返回：
            list: 包含每个信号在指定频率范围内的平均功率谱密度的列表。
        """
        psd_values = []
        for signal in data:    # 按顺序每次取一个通道的数据进行功率谱密度的计算
            f, Pxx = welch(x=signal, fs=fs, nperseg=nperseg, noverlap=int(nperseg / 2))    # 该函数返回两个数组：频率数组和估计的功率谱密度。
            start_index = np.argmax(f >= low)      # 返回频率数组中第一个大于或等于指定低频率的索引
            end_index = np.argmax(f >= high)        # 返回率数组中第一个大于指定高频率的索引
            avg_psd = np.mean(Pxx[start_index:end_index])      # 获取制定频段的功率谱，并计算平均值
            psd_values.append(avg_psd)              # 将计算的平均功率谱添加到psd_values中
        return psd_values
    def trans_alternative_data(self, raw_psd_data, channel_mapping: dict,
                               EEG_name_list: list, alternative_name_list: list):
        """
        将原始PSD数据转换成按照绘图导联顺序重构的数据。

        参数：
            raw_psd_data (numpy.ndarray): 原始PSD数据（一维数组）。[59,]
            channel_mapping (dict): 采集设备的导联顺序映射。
            EEG_name_list (list): 绘图的导联顺序列表。
            alternative_name_list (list): 可选的绘图导联列表。

        返回：
            numpy.ndarray: 处理好的PSD数据。
        """
        reweighted_data = []

        # 复制导联映射，避免修改输入参数
        channel_mapping_copy = channel_mapping.copy()

        # 将到导联与数据对应
        for i, channel in enumerate(EEG_name_list):
            channel_mapping_copy[channel] = raw_psd_data[i]

        # 将不相关导联的对应值全部置零
        for channel in (set(EEG_name_list) - set(alternative_name_list)):
            channel_mapping_copy[channel] = -1

        # 将导联数据按照EEG_name_list重新排序
        for channel in EEG_name_list:
            reweighted_data.append(channel_mapping_copy[channel])

        # 转换为numpy数组并返回处理好的数据
        return np.array(reweighted_data)
    def data_draw(self, data, mode: str = 'ALL',
                  define_name_list=None, fs: int = 1000, nperseg: int = 512):
        """
        绘制脑电图APSD数据处理函数。（三合一）

        参数：
            data (numpy.ndarray): 数据数组。
            quiet (numpy.ndarray): 静息数据数组。
            mode (str): 绘图模式，可选值为 'global'、'C3C4' 或 'self_define'。
            define_name_list (list): 自定义绘图导联列表（仅在 mode 为 'self_define' 时使用）。
            fs (int): 采样频率（默认为 250 Hz）。
            nperseg (int): 窗口长度（默认为 128）。

        返回：
            numpy.ndarray: 绘制的数据数组。
            numpy.ndarray: 通道位置数组。
        """
        psd_array = np.zeros((4, 59))

        for i, (low, high) in enumerate(self.frequent_arrange):
            # 计算每个频段的功率谱密度估计值
            psd_values = self.calculate_avg_power(data=data, low=low, high=high, fs=fs, nperseg=nperseg)
            # 判断处理模式
            if mode == 'ALL':
                trans_name_list = self.EEG_name_list
            elif mode == 'C3C4':
                trans_name_list = self.C3_C4_name_list
            elif mode == '自定义':
                trans_name_list = define_name_list
            else:
                raise ValueError("Invalid mode. Mode must be one of 'global', 'C3C4', or 'self_define'.")
            # 将四个频段的脑电激活度转换为绘图数据并保存在数组中
            psd_array[i, :] = self.trans_alternative_data(raw_psd_data=np.array(psd_values),
                                                          channel_mapping=self.weight,
                                                          EEG_name_list=self.EEG_name_list,
                                                          alternative_name_list=trans_name_list)
        return psd_array, self.ch_pos




