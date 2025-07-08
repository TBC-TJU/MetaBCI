import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.decomposition.sceTRCA import SC_TRCA, FB_SC_TRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references

wp = [(5, 90)]
ws = [(3, 92)]

filterbank = generate_filterbank(wp, ws, srate=250, order=15, rp=0.5)

dataset = Wang2016()
events = dataset.events.keys()
freq_list = [dataset.get_freq(event) for event in events]

Yf = generate_cca_references(freq_list, srate=250, T=0.5, n_harmonics=5)

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],
    srate=250
)


# add 5-90Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(5, 90, l_trans_bandwidth=2, h_trans_bandwidth=5,
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
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

# classifier
filterweights = [(idx_filter + 1) ** (-1.25) + 0.25 for idx_filter in range(5)]
estimator = SC_TRCA(standard=False, ensemble=True, n_components=1)

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    _, p_labels = estimator.fit(X[train_ind], y[train_ind], Yf).predict(X[test_ind])

    accs.append(np.mean(p_labels == y[test_ind]))
print(np.mean(accs))
# If everything is fine, you will get the accuracy about 0.9417.
