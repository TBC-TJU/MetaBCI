# -*- coding: utf-8 -*-
#
# Authors: Duan Shunguo<dsg@tju.edu.cn>
# Date: 2024/9/1
"""
This is a demo for the Bayes algorithm.
"""

from collections import OrderedDict
import numpy as np
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds, 
    generate_loo_indices, match_loo_indices)
from metabci.brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP,
    generate_filterbank, generate_cca_references)
from metabci.brainda.utils.performance import Performance

from metabci.brainda.algorithms.dynamic_stopping import bayes


dataset = Wang2016()
delay = 0.14 
channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
srate = 250 

n_bands = 3
n_harmonics = 5
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]

start_pnt = dataset.events[events[0]][1][0]

paradigm = SSVEP(
    srate=srate, 
    channels=channels, 
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

loo_indices = generate_loo_indices(meta)

# Train the dynamic bayes model
for model_name in models:

    Ds = bayes.Bayes(clone(models[model_name]))
    # It determines the numbers of the train model by the duration of the train data
    for duration in np.round(np.arange(0.2, 1.1, 0.1),2):
        duration = duration 
        Yf = generate_cca_references(
        freqs, srate, duration, 
        phases=None, 
        n_harmonics=n_harmonics)
        print(f"Cunrrunt_model:{model_name},Currunt_Train_Duration: {duration}")
        
        if model_name == 'fbtdca':
            filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+duration))+l]), np.copy(y)
        else:
            filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+duration))]), np.copy(y)
        

        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
    
        train_ind, validate_ind, _ = match_loo_indices(
            5, meta, loo_indices)
        train_ind = np.concatenate([train_ind, validate_ind])

        trainX, trainY = filterX[train_ind], filterY[train_ind]
        Ds.fit(trainX,trainY,duration,Yf)
            
    tlabels = []
    plabels = []
    T_time = []
   
   # Test the dynamic bayes model by trail data once by once 
    for i in range(0,40):
        if model_name == 'fbtdca':
            filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+duration))+l]), np.copy(y)
        else:
            filterX, filterY = np.copy(X[..., int(srate*delay):int(srate*(delay+duration))]), np.copy(y)
        

        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
        _, _, test_ind = match_loo_indices(
                        5, meta, loo_indices)
        testX, testY = filterX[test_ind], filterY[test_ind]

        bufferX = testX[i:i+1,:,:]
        bufferY = testY[i:i+1]
        a = 0.2
        default_duration = round(a,2)
        if model_name == 'fbtdca':
            trail = bufferX[:,:,0:int(srate*default_duration)+l]
        else:
            trail = bufferX[:,:,0:int(srate*default_duration)]
        
        # decide function make the decision by the trail data and output the bool(true or false) and label
        # if the bool is false, the duration of the trail data will be increased by 0.1s and continue to decide
        bool,label = Ds.predict(trail,default_duration,1)
        while not bool:
            a += 0.1
            default_duration = round(a,2)
            if model_name == 'fbtdca':
                trail = bufferX[:,:,0:int(srate*default_duration)+l]
            else:
                trail = bufferX[:,:,0:int(srate*default_duration)]
            bool,label = Ds.predict(trail,default_duration,1)
        
        tlabels.append(bufferY[0])
        plabels.append(label)
        T_time.append(default_duration)


    performance = Performance(estimators_list=["Acc","tITR"],Tw=np.mean(T_time))
    results = performance.evaluate(y_true=np.array(tlabels),y_pred=np.array(plabels))
    print(f"Model: {model_name}, results: {results}")