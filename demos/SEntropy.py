
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds, generate_kfold_indices, match_kfold_indices)
from metabci.brainda.algorithms.\
    feature_analysis.nolinear_dynamic_analysis import SEntropy
from metabci.brainda.algorithms.decomposition.base import generate_filterbank


wp = [(4, 8), (8, 12), (12, 30)]
ws = [(2, 10), (6, 14), (10, 32)]
filterbank = generate_filterbank(wp, ws, srate=128, order=4, rp=0.5)

dataset = AlexMI()
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=[(0, 3)],  # 3 seconds
    srate=128)

# Add 6-30Hz bandpass filter in raw hook.


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


chan_names = AlexMI()._CHANNELS
srate = 128
chan_names[0] = 'Fpz'
chan_names[3] = 'Fz'
chan_names[12] = 'Pz'

sentropy = SEntropy(deaverage=False, order=2, delay=1, sfreq=srate,
                    figsize=(10, 4), chan_names=chan_names, headsize=0.05,
                    cmap='Reds', fontsize_title=20, fontweight_title='bold',
                    fontcolor_title='black', fontstyle_title='normal',
                    loc_title='center', pad_title=15, fontsize_colorbar=20,
                    fontweight_colorbar='bold', fontcolor_colorbar='black',
                    fontstyle_colorbar='normal', label_colorbar='SEntropy',
                    loc_colorbar='center')
sentropy.fit(X, y)
X_entropy = sentropy.transform(X)
sentropy.draw()


# 5-fold cross validation
set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# SEntropy with SVC classifier
estimator = make_pipeline(*[
    SEntropy(deaverage=False, order=2, delay=1),
    SVC()
])

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels == y[test_ind]))
print(np.mean(accs))
