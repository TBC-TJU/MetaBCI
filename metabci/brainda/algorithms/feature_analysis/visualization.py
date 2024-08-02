# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.signal import welch, spectrogram
import mne

class MetaBCIVisualization:
    def __init__(self, data=None, fs=None):
        self.data = data
        self.fs = fs

    # 混淆矩阵可视化
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def evaluate_classification(self, y_true, y_pred, labels):
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels)
        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)
        self.plot_confusion_matrix(y_true, y_pred, labels)

    # 频谱密度图可视化
    def plot_power_spectral_density(self):
        if self.data is None or self.fs is None:
            raise ValueError("Data and sampling frequency must be provided for PSD visualization.")
        
        plt.figure(figsize=(10, 6))
        for i in range(self.data.shape[0]):
            f, Pxx = welch(self.data[i], self.fs, nperseg=1024)
            plt.semilogy(f, Pxx, label=f'Channel {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2/Hz)')
        plt.title('Power Spectral Density')
        plt.legend()
        plt.show()

    # 伪彩色图可视化
    def plot_pseudocolor(self, xlabel='Time (samples)', ylabel='Channels', title='EEG Pseudocolor Plot'):
        if self.data is None:
            raise ValueError("Data must be provided for pseudocolor plot visualization.")
        
        plt.figure(figsize=(12, 6))
        plt.imshow(self.data, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='EEG Amplitude (μV)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    # 时频分析可视化
    def plot_time_frequency(self):
        if self.data is None or self.fs is None:
            raise ValueError("Data and sampling frequency must be provided for time-frequency visualization.")
        
        plt.figure(figsize=(12, 6))
        for i in range(self.data.shape[0]):
            f, t, Sxx = spectrogram(self.data[i], self.fs)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.title(f'Time-Frequency Analysis (Channel {i+1})')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.show()

    # 绘制时域波形图
    def plot_single_trial(self, data, sample_num, axes=None, amp_mark=False, time_start=0, time_end=1):
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='EEG Signal')
        if amp_mark:
            max_amplitude = np.max(data[time_start:time_end])
            max_index = np.argmax(data[time_start:time_end])
            plt.plot(max_index + time_start, max_amplitude, 'ro')  # Mark the peak
            plt.annotate(f'Peak: {max_amplitude}', xy=(max_index + time_start, max_amplitude), xytext=(max_index + time_start + 5, max_amplitude + 0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05))
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Single Trial EEG Signal')
        plt.legend()
        plt.show()

    # 绘制频域波形图
    def plot_multi_trials(self, data, sample_num, axes=None):
        plt.figure(figsize=(10, 5))
        for i in range(data.shape[0]):
            plt.plot(data[i], label=f'Trial {i+1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Multiple Trials EEG Signal')
        plt.legend()
        plt.show()

    # 绘制脑地形图
    def plot_topomap(self, data, ch_names, fig, point=0, srate=-1, ch_types='eeg', axes=None):
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        evoked = mne.EvokedArray(data, info)
        evoked.plot_topomap(times=point / srate, size=3, title='EEG Topomap', axes=axes)
        plt.show()

# # 使用示例
# if __name__ == "__main__":
#     data = np.random.rand(10, 1000)  # 10个通道的示例数据
#     fs = 1000  # 采样率
#     y_true = [0, 1, 2, 2, 0, 1, 0, 2, 1, 1]  # 示例真实标签
#     y_pred = [0, 2, 2, 2, 0, 0, 0, 1, 1, 1]  # 示例预测标签
#     labels = ['Class 0', 'Class 1', 'Class 2']

#     visualization = MetaBCIVisualization(data, fs)
#     visualization.plot_pseudocolor()
#     visualization.plot_power_spectral_density()
#     visualization.plot_time_frequency()
#     visualization.evaluate_classification(y_true, y_pred, labels)
#     visualization.plot_single_trial(data[0], sample_num=1000, amp_mark=True, time_start=200, time_end=800)
#     visualization.plot_multi_trials(data, sample_num=1000)

#     # 绘制脑地形图示例
#     ch_names = ['Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
#     fig = plt.figure()
#     visualization.plot_topomap(data[0], ch_names, fig, point=500, srate=fs)
