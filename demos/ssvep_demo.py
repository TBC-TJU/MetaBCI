# -*- coding: utf-8 -*-
# SSVEP Classification Demo

from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score

from brainda.datasets import Wang2016
from brainda.paradigms import SSVEP
from brainda.algorithms.utils.model_selection import (
    set_random_seeds, 
    generate_loo_indices, match_loo_indices)
from brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP,
    generate_filterbank, generate_cca_references)


dataset = Wang2016()
delay = 0.14 # seconds
channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
srate = 250 # Hz
duration = 0.5 # seconds
n_bands = 3
n_harmonics = 5
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]

Yf = generate_cca_references(
    freqs, srate, duration, 
    phases=None, 
    n_harmonics=n_harmonics)

start_pnt = dataset.events[events[0]][1][0]
paradigm = SSVEP(
    srate=srate, 
    channels=channels, 
    intervals=[(start_pnt+delay, start_pnt+delay+duration+0.1)], # more seconds for TDCA 
    events=events)

wp = [[8*i, 90] for i in range(1, n_bands+1)]
ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
filterbank = generate_filterbank(
    wp, ws, srate, order=4, rp=1)
filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25

def data_hook(X, y, meta, caches):
    filterbank = generate_filterbank(
        [[8, 90]], [[6, 95]], srate, order=4, rp=1)
    X = sosfiltfilt(filterbank[0], X, axis=-1)
    return X, y, meta, caches

paradigm.register_data_hook(data_hook)

set_random_seeds(64)
l = 5
models = OrderedDict([
    ('fbscca', FBSCCA(
            filterbank, filterweights=filterweights)),
    ('fbecca', FBECCA(
            filterbank, filterweights=filterweights)),
    ('fbdsp', FBDSP(
            filterbank, filterweights=filterweights)),
    ('fbtrca', FBTRCA(
            filterbank, filterweights=filterweights)),
    ('fbtdca', FBTDCA(
            filterbank, l, n_components=8, 
            filterweights=filterweights)),
])

X, y, meta = paradigm.get_data(
    dataset, 
    subjects=[1], 
    return_concat=True, 
    n_jobs=1, 
    verbose=False)

set_random_seeds(42)
loo_indices = generate_loo_indices(meta)

for model_name in models:
    if model_name == 'fbtdca':
        filterX, filterY = np.copy(X[..., :int(srate*duration)+l]), np.copy(y)
    else:
        filterX, filterY = np.copy(X[..., :int(srate*duration)]), np.copy(y)
    
    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

    n_loo = len(loo_indices[1][events[0]])
    loo_accs = []
    for k in range(n_loo):
        train_ind, validate_ind, test_ind = match_loo_indices(
            k, meta, loo_indices)
        train_ind = np.concatenate([train_ind, validate_ind])

        trainX, trainY = filterX[train_ind], filterY[train_ind]
        testX, testY = filterX[test_ind], filterY[test_ind]

        model = clone(models[model_name]).fit(
            trainX, trainY,
            Yf=Yf
        )
        pred_labels = model.predict(testX)
        loo_accs.append(
            balanced_accuracy_score(testY, pred_labels))
    print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))


