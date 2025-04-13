import numpy as np
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from scipy import signal
from metabci.brainda.algorithms.\
    feature_analysis.time_freq_analysis import CWT
import matplotlib.pyplot as plt


wp = [(4, 8), (8, 12), (12, 30)]
ws = [(2, 10), (6, 14), (10, 32)]
filterbank = generate_filterbank(wp, ws, srate=128, order=4, rp=0.5)


dataset = AlexMI()
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=[(0, 3)],  # 3 seconds
    srate=128
)

# add 6-30Hz bandpass filter in raw hook


def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def epochs_hook(epochs, caches):
    # do something with epochs object
    print(epochs.event_id)
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches


def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
    print("Raw stage:{},Epochs stage:{}".format(
        caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches


paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)


X, y, meta = paradigm.get_data(
    dataset,
    subjects=[3],
    return_concat=True,
    n_jobs=None,
    verbose=False)

srate = 128
index = [y == label for label in np.unique(y)]
# Average the trials for each class
X_class0 = np.mean(X[index[0]], axis=0, keepdims=True)
X_class1 = np.mean(X[index[1]], axis=0, keepdims=True)
print(dataset._CHANNELS)
channel_id_class0 = dataset._CHANNELS.index('C3')
channel_id_class1 = dataset._CHANNELS.index('PZ')
freq = np.linspace(1, srate / 2, 100)

plt.figure()
cwt = CWT(n_jobs=None, fs=srate, wavelet=signal.morlet2, freqs=freq, omega0=5,
          dtype="complex128", trail_id=0, channel_id=channel_id_class0,
          cmap='summer', vmin=None, vmax=None,
          fontsize_title=12, fontweight_title='bold',
          fontcolor_title='black', fontstyle_title='oblique',
          title="Class0: Complex Morlet Wavelet Transform of 'C3'",
          distance=0.1)
cwt.fit(X_class0)
CWT_complexmatrix_class0, CWT_spectrum_energy_class0 = cwt.transform(X_class0)
cwt.draw()

plt.figure()
cwt = CWT(n_jobs=None, fs=srate, wavelet=signal.morlet2, freqs=freq, omega0=5,
          dtype="complex128", trail_id=0, channel_id=channel_id_class1,
          cmap='summer', vmin=None, vmax=None,
          fontsize_title=12, fontweight_title='bold',
          fontcolor_title='black', fontstyle_title='oblique',
          title="Class1: Complex Morlet Wavelet Transform of 'Pz'",
          distance=0.1)
cwt.fit(X_class1)
CWT_complexmatrix_class1, CWT_spectrum_energy_class1 = cwt.transform(X_class1)
cwt.draw()
