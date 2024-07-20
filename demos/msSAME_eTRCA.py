import sys
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_shuffle_indices, match_shuffle_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.transfer_learning import SAME, MSSAME


wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]

filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)

dataset = Wang2016()
a = [np.round(i,1) for i in np.arange(8,16,0.2)]
events = [str(int(i)) if i % 1==0 else str(i) for i in a]
# '8', '8.2', '8.4', ..., '15.6', '15.8'
freq_list = [dataset.get_freq(event) for event in events]
phase_list = [dataset.get_phase(event) for event in events]

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],  # Tw = 0.5
    srate=250,events=events  #  events: Frequency ascending order
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
indices = generate_shuffle_indices(meta, n_splits=n_splits,train_size=1,validate_size=1,test_size=4)  # Nt = 2

# classifier
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
# estimator  = FBTRCA(filterbank=filterbank,n_components = 1, ensemble = True,filterweights=np.array(filterweights), n_jobs=-1)

accs = []
accs_withSAME = []
accs_withmsSAME = []
for k in range(n_splits):

    train_ind, validate_ind, test_ind = match_shuffle_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    del validate_ind
    # get data and label for training and testing
    X_train , y_train = X[train_ind], y[train_ind]
    X_test  , y_test =  X[test_ind],  y[test_ind]

    # count acc (without SAME)
    estimator = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights),
                       n_jobs=-1)
    p_labels = estimator.fit(X_train , y_train).predict(X_test)  # without SAME
    accs.append(np.mean(p_labels==y_test))

    # count acc (with SAME)
    same = SAME(fs = 250, Nh = 5, flist = freq_list, n_Aug = 4)
    same.fit(X_train , y_train)
    X_aug1, y_aug1 = same.augment()
    X_train_new1 = np.concatenate((X_train, X_aug1), axis=0)
    y_train_new1 = np.concatenate((y_train, y_aug1), axis=0)
    estimator = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights),
                       n_jobs=-1)
    p_labels_withSAME = estimator.fit(X_train_new1, y_train_new1).predict(X_test)  # with SAME
    accs_withSAME.append(np.mean(p_labels_withSAME == y_test))

    # count acc (with msSAME)
    mssame = MSSAME(fs = 250, Nh = 5, flist = freq_list, plist=phase_list, n_Aug=4, n_Neig=12) # When n_Neig=1, the result is similar to SAME
    mssame.fit(X_train , y_train)
    X_aug2, y_aug2 = mssame.augment()
    X_train_new2 = np.concatenate((X_train, X_aug2), axis=0)
    y_train_new2 = np.concatenate((y_train, y_aug2), axis=0)
    estimator = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights),
                       n_jobs=-1)
    p_labels_withmsSAME = estimator.fit(X_train_new2, y_train_new2).predict(X_test)  # with msSAME
    accs_withmsSAME.append(np.mean(p_labels_withmsSAME == y_test))

print('without SAME',np.mean(accs))
print('with SAME',np.mean(accs_withSAME))
print('with msSAME',np.mean(accs_withmsSAME))


# If everything is fine, you will get the accuracy about:
# without SAME:  0.578125
# with SAME:     0.777083
# with msSAME:   0.828125

