import os
import zipfile
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
from collections import Counter

import numpy as np
from mne import create_info
from mne.io import RawArray, Raw
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
import scipy.io


class Experiment(BaseDataset):

    def __init__(self, path=None, experiment_name='training'):
        # Eg. path = 'C:\\Users\\abc\\AssistBCI'
        '''
        Experiment将接受buffer（字典类型）输入，buffer中包含如:
                location, experiment_name, paradigm, n_elements,
                epoch_ind, subject, channels, frequency, phases 等参数
        '''

        if path == None:
            user_home = os.path.expanduser('~')
            user_dir = os.path.join(user_home, 'AssistBCI\\Experiment_Raw_data')
            info_dir = os.path.join(user_home, 'AssistBCI\\Experiment_Raw_data_info')
            if not os.path.exists(user_dir) or not os.path.exists(info_dir):
                raise (FileNotFoundError(path))
        else:
            user_dir = os.path.join(path, 'AssistBCI\\Experiment_Raw_data')
            info_dir = os.path.join(path, 'AssistBCI\\Experiment_Raw_data_info')
            if not os.path.exists(user_dir) or not os.path.exists(info_dir):
                raise (FileNotFoundError(path))

        self._data_paths = []

        data_paths = [os.path.join(user_dir, file) for file in os.listdir(user_dir)]
        if not data_paths:
            raise (FileNotFoundError(path))

        for data_path in data_paths:
            if data_path.split('\\')[-1].split('_')[1] == experiment_name:
                self._data_paths.append('file://' + data_path.replace('\\', '/'))

        # self.location = "file://C:/Users/m1358/Desktop/test/"

        file_path_info = info_dir + '\\' + 'E_' + experiment_name + '.txt'
        try:
            # 读取txt文件
            with open(file_path_info, 'r') as f:
                lines = f.readlines()
                # 解析文件内容
                stim_info = {}
                for line in lines:
                    elements = line.split('\n')[0]
                    elements = elements.split(':')
                    try:
                        stim_info[elements[0]] = eval(':'.join(elements[1:]))
                    except:
                        stim_info[elements[0]] = ':'.join(elements[1:])

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")


        print(stim_info)

        self.experiment_name = experiment_name

        self.paradigm = stim_info['paradigm'].lower()

        self.n_elements = stim_info['n_elements']

        self.srate = stim_info['srate']

        self.epoch_ind = [140, 5140]

        self.epoch_ind_time = [i / self.srate for i in self.epoch_ind]

        self.subjects = stim_info['subject']

        self.events_name = [str(name) for name in stim_info['freqs']]

        self._CHANNELS = stim_info['channels']

        self._FREQS = stim_info['freqs']

        self._PHASES = stim_info['phases']  # 指令的相位

        self._EVENTS = {event_name:
                            (i+1, (self.epoch_ind_time[0], self.epoch_ind_time[1]))
                        for i, event_name in enumerate(self.events_name)}


        super().__init__(
            dataset_code=self.experiment_name,
            subjects=self.subjects,
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=self.srate,
            paradigm=self.paradigm,
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

        subject = int(subject)

        self._data_paths.sort(reverse=True)

        for data_path in self._data_paths:
            if data_path.split('/')[-1].split('_')[3] == str(subject):
                url = data_path
                break

        file_dest = mne_data_path(
            url,
            self.experiment_name,
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )

        return file_dest

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        raw_mat = scipy.io.loadmat(dests)

        trail_id = []
        for trail in raw_mat:
            if trail[0].isdigit():
                trail_id.extend([int(trail.split("_")[0])])
        count = Counter(trail_id)
        if len(set(list(count.values()))) != 1:
            raise (ValueError("Invalid Experiment Dataset: trial loose in block"))

        trail_id = []
        trail_data = []
        epoch_data = []
        for trail in raw_mat:
            if trail[0].isdigit():
                id = int(trail.split("_")[0])

                if id in trail_id:

                    try:
                        epoch_data = np.concatenate((epoch_data, trail_data[:, :, :, np.newaxis]),
                                                    axis=3)
                    except:
                        epoch_data = trail_data[:, :, :, np.newaxis]

                    trail_id = []
                    trail_data = []

                trail_id.extend([id])
                data = np.append(
                    np.array(raw_mat[trail]).T[0:8, :],
                    #np.array(raw_mat[trail]).T,
                    np.zeros([1, len(raw_mat[trail])]),
                    axis=0
                )
                data[-1, self.epoch_ind[0]] = id

                try:
                    trail_data = np.concatenate((trail_data, data[:, :, np.newaxis]), axis=2)
                except:
                    trail_data = data[:, :, np.newaxis]

        epoch_data = np.concatenate((epoch_data, trail_data[:, :, :, np.newaxis]),
                                    axis=3)

        data = np.transpose(epoch_data, (0, 3, 2, 1))

        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        ch_names = [ch_name.upper() for ch_name in self._CHANNELS]
        ch_names = ch_names + ["STI 014"]
        ch_types = ["eeg"] * 9
        ch_types[-1] = "stim"

        info = create_info(ch_names=ch_names,
                           ch_types=ch_types, sfreq=self.srate)

        runs = dict()
        for i in range(data.shape[1]):
            raw = RawArray(
                data=np.reshape(data[:, i, ...],
                                (data.shape[0], -1)),
                info=info
            )
            raw.set_montage(montage)
            runs["run_{:d}".format(i)] = raw

        sess = {"session_0": runs}
        return sess

    def get_freq(self, event: str):
        return self._FREQS[self._EVENTS[event][0] - 1]

    def get_phase(self, event: str):
        return self._PHASES[self._EVENTS[event][0] - 1]