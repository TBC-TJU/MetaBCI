from typing import Union, Optional, Dict, List
from pathlib import Path
from mne.io import read_raw_cnt, Raw
from mne.channels import make_standard_montage
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.utils.channels import upper_ch_names


class TffMiData(BaseDataset):
    __EVENTS = dict(left_hand=(1, (0, 4)),
                    right_hand=(2, (0, 4))
                    )
    __CHANNELS = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ',
                  'C4', 'CP3', 'CPZ', 'CP4']

    #   1.cnt 2.cnt [1,2]
    __RUNS = list(range(1, 3))

    #   baseurl\\sub{:d}\\{:d}.cnt'
    __BASE_URL = "data\\mi"

    def __init__(self, subjects, sample_rate, paradigm='imagery'):
        super().__init__(
            dataset_code="TFF",
            subjects=subjects,
            events=self.__EVENTS,
            channels=self.__CHANNELS,
            srate=sample_rate,
            paradigm=paradigm
        )

    def data_path(self,
                  subject: Union[str, int],
                  path: Optional[Union[str, Path]] = None,
                  force_update: bool = False,
                  update_path: Optional[bool] = None,
                  proxies: Optional[Dict[str, str]] = None,
                  verbose: Optional[Union[bool, str, int]] = None
                  ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        subjects_file_list = []
        for sub in self.subjects:
            subjects_file_list.append([Path('{:s}\\sub{}\\{:d}.cnt'.format(
                self.__BASE_URL, sub, run)) for run in self.__RUNS])
        return subjects_file_list

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        subjects_file_list = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        sess = dict()
        for isess, run_dests in enumerate(subjects_file_list):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_cnt(run_file,
                                   eog=['HEO', 'VEO'],
                                   ecg=['EKG'], emg=['EMG'],
                                   misc=[32, 42, 59, 63],
                                   preload=True)
                raw = upper_ch_names(raw)
                raw = raw.pick_types(eeg=True, stim=True,
                                     selection=self.channels)
                raw.set_montage(montage)

                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess


