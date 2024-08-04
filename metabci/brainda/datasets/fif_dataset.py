import os
import mne
from mne.io import read_raw_fif, RawArray,base,Raw
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.utils.channels import upper_ch_names
from mne import create_info
from mne.channels import make_standard_montage
from metabci.brainda.utils.download import mne_data_path
from pathlib import Path
from typing import Union, Optional, Dict, List,cast
import numpy as np

test1_URL = r'path\to\your\data'

class Test1(BaseDataset):

    _CHANNELS = ['TP10', 'O2', 'OZ', 'O1', 'POZ', 'PZ', 'TP9', 'FCZ','TRIGGER']

    _FREQS = [
        8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6,
        12.0, 12.4, 12.8, 13.2, 13.6, 14.0, 14.4, 14.8, 15.2, 15.6
    ]

    _PHASES = [
        0.0, 0.35, 0.7, 1.05, 1.4, 1.75, 0.1, 0.45, 0.8, 1.15,
        1.5, 1.85, 0.2, 0.55, 0.9, 1.25, 1.6, 1.95, 0.3, 0.65
    ]
    # 更新 _EVENTS 字典
    _EVENTS = {}
    for i, freq in enumerate(_FREQS):
        if freq == 12.8:
            _EVENTS[str(freq)] = (21, (0, 1))
        elif i + 1 != 13:
            _EVENTS[str(freq)] = (i + 1, (0, 1))
    def __init__(self):
        super().__init__(
            dataset_code="test1",
            subjects=list(range(1, 1000)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=500,
            paradigm="ssvep"
        )

    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        subject = cast(int, subject)
        file_name = "{:d}.fif".format(subject)
        file_dest = os.path.join(test1_URL, file_name)
        dests = [[file_dest]]
        return dests

    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        raw_eeg = mne.io.read_raw_fif(dests[0][0], preload=True, verbose=verbose)
        #montage
        montage = make_standard_montage("standard_1020")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # 检查通道名称
        expected_ch_names = [ch_name.upper() for ch_name in self._CHANNELS]
        actual_ch_names = raw_eeg.ch_names

        # 打印通道名称不匹配的信息
        missing_channels = set(expected_ch_names) - set(actual_ch_names)
        extra_channels = set(actual_ch_names) - set(expected_ch_names)
        if missing_channels or extra_channels:
            print(f"Channel name mismatch detected:")
            if missing_channels:
                print(f"Missing channels: {missing_channels}")
            if extra_channels:
                print(f"Extra channels: {extra_channels}")

        # 检查通道类型
        channel_types = {'TP10': 'eeg', 'O2': 'eeg', 'OZ': 'eeg', 'O1': 'eeg', 'POZ': 'eeg',
                         'PZ': 'eeg', 'TP9': 'eeg', 'FCZ': 'eeg', 'TRIGGER': 'stim'}

        # 打印通道类型不匹配的信息
        mismatched_channels = []
        for ch_name in raw_eeg.ch_names:
            if raw_eeg.get_channel_types(ch_name) != channel_types.get(ch_name):
                mismatched_channels.append((ch_name, raw_eeg.get_channel_types(ch_name)))

        if mismatched_channels:
            print(f"Channel type mismatches detected:")
            for ch_name, actual_type in mismatched_channels:
                print(f"{ch_name}: Expected '{channel_types[ch_name]}', found '{actual_type}'")
        raw_eeg.set_montage(montage)
    # 转为 raw 格式
        sess = {"session_0": {"run_0": raw_eeg}}
        return sess

    def get_freq(self, event: str):
        # 获取事件ID
        event_id = self._EVENTS[event][0]

        # 如果事件ID为21（即频率为12.8Hz的情况），则使用特殊处理
        if event_id == 21:
            return 12.8
        else:
            # 否则按正常逻辑计算
            return self._FREQS[event_id - 1]

    def get_phase(self, event: str):
        # 获取事件ID
        event_id = self._EVENTS[event][0]

        # 如果事件ID为21（即频率为12.8Hz的情况），则使用特殊处理
        if event_id == 21:
            return self._PHASES[12]  # 因为12.8Hz对应的索引为12
        else:
            # 否则按正常逻辑计算
            return self._PHASES[event_id - 1]
        
        
        
        
'''dataset = Test1()
file_paths = dataset.data_path(subject=1)
print(file_paths)
subject_data = dataset._get_single_subject_data(1)
print(subject_data)
for event, (event_id, _) in dataset._EVENTS.items():
    freq = dataset.get_freq(event)
    phase = dataset.get_phase(event)
    print(f"Event ID: {event_id} (Frequency: {freq} Hz, Phase: {phase})")'''
