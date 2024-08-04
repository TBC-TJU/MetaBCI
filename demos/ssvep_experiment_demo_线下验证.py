# -*- coding: utf-8 -*-
# SSVEP Classification Demo
import pickle
from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import kurtosis

from metabci.brainda.datasets import Experiment
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices, match_loo_indices)
from metabci.brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP, TRCA, TRCAR,
    generate_filterbank, generate_cca_references)

subject = 0

dataset = Experiment(experiment_name='Training')
channels = dataset.channels
# channels = ['PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
srate = dataset.srate  # Hz

delay = 0  # seconds
duration = 4  # seconds
n_bands = 5
n_harmonics = 4

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
    ('trca', TRCA(n_components=1)),
    ('trcar', TRCAR(n_components=1))
])

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[subject],
    return_concat=True,
    n_jobs=1,
    verbose=False)

set_random_seeds(42)
loo_indices = generate_loo_indices(meta)

for i, model_name in enumerate(models):
    if model_name == 'fbtdca':
        filterX, filterY = np.copy(X[..., :int(srate * duration) + l]), np.copy(y)
    else:
        filterX, filterY = np.copy(X[..., :int(srate * duration)]), np.copy(y)

    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

    n_loo = len(loo_indices[subject][events[0]])
    loo_accs = []
    for k in range(n_loo):
        train_ind, validate_ind, test_ind = match_loo_indices(
            k, meta, loo_indices)
        train_ind = np.concatenate([train_ind, validate_ind])

        trainX, trainY = filterX[train_ind], filterY[train_ind],
        testX, testY = filterX[test_ind], filterY[test_ind]
        # Yf, Yf_Y = filterX[validate_ind], filterY[validate_ind]
        # print("Yf_Y:", Yf_Y)
        model = clone(models[model_name]).fit(
            trainX, trainY,
            Yf=Yf
        )
        pred_labels, features = model.predict(testX)
        print("labels:", pred_labels)
        print("kurtosis", kurtosis(np.sort(features), axis=-1,fisher=False))
        loo_accs.append(
            balanced_accuracy_score(testY, pred_labels))

        # import os
        # user_dir = os.path.join(os.path.expanduser('~'), 'AssistBCI\\Personal_Model')
        # if not os.path.exists(user_dir):
        #     os.makedirs(user_dir)
        #
        # import datetime
        #
        # name = model_name + datetime.datetime.now().strftime("_%Y%m%d%H%M%S")
        # model_dir = os.path.join(user_dir, name + '.pkl')
        #
        #
        # with open(model_dir, 'wb') as file:
        #     pickle.dump(model, file)
        #
        # print("saved")
        #
        # del model
        #
        # with open(model_dir, 'rb') as file:
        #     new_model = pickle.load(file)
        #
        # pred_labels, features = new_model.predict(testX)
        # print("new_model labels:", pred_labels)
        # print("new_model kurtosis", kurtosis(np.sort(features), axis=-1, fisher=False))



    print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))


