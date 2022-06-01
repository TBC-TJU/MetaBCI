import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import FBCSP
from brainda.algorithms.decomposition.base import generate_filterbank

wp=[(4,8),(8,12),(12,30)]
ws=[(2,10),(6,14),(10,32)]
filterbank = generate_filterbank(wp,ws,srate=128,order=4,rp=0.5)

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
    subjects=[3],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# 5-fold cross validation
set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# FBCSP with SVC classifier
estimator = make_pipeline(*[
    FBCSP(n_components=5,n_mutualinfo_components=4,filterbank=filterbank),
    SVC()
])

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels==y[test_ind]))
print(np.mean(accs))
# If everything is fine, you will get the accuracy about 0.875.