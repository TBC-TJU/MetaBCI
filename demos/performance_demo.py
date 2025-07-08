import sys
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_shuffle_indices, match_shuffle_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.utils.performance import Performance

wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]

filterbank = generate_filterbank(wp, ws, srate=250, order=15, rp=0.5)

dataset = Wang2016()

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
n_splits = 6
# train and validate set will be merged
indices = generate_shuffle_indices(meta, n_splits=n_splits, train_size=2, validate_size=1, test_size=3)  # Nt = 3

# classifier
filterweights = [(idx_filter + 1) ** (-1.25) + 0.25 for idx_filter in range(5)]
estimator = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights),
                   n_jobs=-1)
# performance: accuracy, practical ITR, TPR, AUC
performance = Performance(estimators_list=["Acc","pITR","TPR","AUC"], Tw=0.5, Ts=0.5)

k = 0  # the index of k-th splits

train_ind, validate_ind, test_ind = match_shuffle_indices(k, meta, indices)
# merge train and validate set
train_ind = np.concatenate((train_ind, validate_ind))
# train and test
p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
p_corr = estimator.fit(X[train_ind], y[train_ind]).transform(X[test_ind])

# performance evaluate
results = performance.evaluate(y_true=y[test_ind], y_pred=p_labels, y_score=p_corr)
print(results)

# if everything is ok, you will get following results:
# {'Acc': 0.9, 'pITR': 259.4635367647114, 'TPR': 0.9, 'AUC': 0.9978632478632479}
