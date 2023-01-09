import sys
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_shuffle_indices, match_shuffle_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.transfer_learning import SAME


wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]

filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)

dataset = Wang2016()
events = dataset.events.keys()
freq_list = [dataset.get_freq(event) for event in events]

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],  # Tw = 0.5
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

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[23],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# cross validation
set_random_seeds(38)
n_splits = 6
# train and validate set will be merged
indices = generate_shuffle_indices(meta, n_splits=n_splits,train_size=2,validate_size=1,test_size=3)  # Nt = 3

# classifier
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
estimator = FBTRCA(filterbank=filterbank,n_components = 1, ensemble = True,filterweights=np.array(filterweights), n_jobs=-1)


accs = []
accs_withSAME = []
for k in range(n_splits):
    train_ind, validate_ind, test_ind = match_shuffle_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    X_train , y_train = X[train_ind], y[train_ind]
    # SAME
    same = SAME(fs = 250, Nh = 5, flist = freq_list, n_Aug = 4)
    same.fit(X_train , y_train)
    X_aug, y_aug = same.augment()
    X_train_new = np.concatenate((X_train, X_aug), axis=0)
    y_train_new = np.concatenate((y_train, y_aug), axis=0)
    # count acc
    p_labels = estimator.fit(X_train , y_train).predict(X[test_ind])  # without SAME
    p_labels_withSAME = estimator.fit(X_train_new, y_train_new).predict(X[test_ind])  # with SAME
    accs.append(np.mean(p_labels==y[test_ind]))
    accs_withSAME.append(np.mean(p_labels_withSAME == y[test_ind]))
print('withoutSAME',np.mean(accs))
print('withSAME',np.mean(accs_withSAME))
# If everything is fine, you will get the accuracy about:
# withoutSAME:  0.7291666
# withSAME:     0.8236111

