import sys
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.transfer_learning import LST
from metabci.brainda.algorithms.decomposition.base import generate_filterbank


wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]

filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)

dataset = Wang2016()

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],
    srate=250
)

# add 5-90Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(5, 90, l_trans_bandwidth=2,h_trans_bandwidth=5,
        phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

def epochs_hook(epochs, caches):
    # do something with epochs object
    # print(epochs.event_id)
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches

def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
    # print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches

paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

# target_subject
Xt, yt, meta_t = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# source_subject
Xs, ys, meta_s = paradigm.get_data(
    dataset,
    subjects=[2],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta_t, kfold=kfold)

# classifier
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
estimator = FBTRCA(filterbank=filterbank,n_components = 1, ensemble = True,filterweights=np.array(filterweights), n_jobs=-1)


accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta_t, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    # transfer_learning
    LST_estimator = LST()
    LST_estimator.fit(Xt[train_ind], yt[train_ind])
    Xs_transform = LST_estimator.transform(Xs, ys)
    Xnew = np.concatenate((Xt[train_ind], Xs_transform), axis=0)
    ynew = np.concatenate((yt[train_ind], ys), axis=0)
    # train and test
    p_labels = estimator.fit(Xnew, ynew).predict(Xt[test_ind])
    accs.append(np.mean(p_labels==yt[test_ind]))
print(np.mean(accs))


