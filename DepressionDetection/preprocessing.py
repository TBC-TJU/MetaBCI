# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
from mne.preprocessing import ICA
from mne import pick_types

def preprocess_data(raw):
    # 滤波
    raw.filter(1., 40., fir_design='firwin')

    # 重参考
    raw.set_eeg_reference('average', projection=True)

    # ICA 去伪迹
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.apply(raw)

    return raw
