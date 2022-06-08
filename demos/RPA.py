import sys
import numpy as np
from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.manifold import get_recenter, recenter, get_rescale, rescale, get_rotate, rotate

dataset = AlexMI()
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=[(0, 3)], # 3 seconds
    srate=128
)

# add 6-30Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2,h_trans_bandwidth=5,
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
    print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches

paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1,2],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# source data
ind_s = meta [meta[ 'subject' ]== 1].index.to_numpy()
Xs, ys = X[ind_s], y[ind_s]

# target data
ind_t = meta [meta[ 'subject' ]== 2].index.to_numpy()
Xt, yt = X[ind_t], y[ind_t]

# Re-Center Matrices (unsupervised)
iM12 = get_recenter(Xs, mean_method= 'riemann')
Xs = recenter(Xs,iM12)
iM12 = get_recenter(Xt, mean_method= 'riemann')
Xt = recenter(Xt,iM12)

# Equalize the Dispersions (unsupervised)
M, scale = get_rescale(Xs)
Xs = rescale(Xs,M, scale)
M, scale = get_rescale(Xt)
Xt = rescale(Xt,M, scale)

# Rotate Around the Geometric Mean (supervised)
Ropt = get_rotate(Xs, ys, Xt, yt)
Xt_final = rotate(Xt,Ropt)