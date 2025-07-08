import sys
import numpy as np
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sLDA
from metabci.brainda.algorithms.transfer_learning import MEKT


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

# target_subject
Xt, yt, meta_t = paradigm.get_data(
    dataset,
    subjects=[3],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# soure_subjects
Xs, ys, meta_s = paradigm.get_data(
    dataset,
    subjects=[1,2,4,5],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# sLDA classifier
estimator = sLDA(solver='lsqr', shrinkage='auto')
# MEKT transfer learning
mekt = MEKT(max_iter=5, covariance_type='lwf')

source_features, target_features = mekt.fit_transform(Xs, ys, Xt)
p_labels = estimator.fit(source_features, ys).predict(target_features)
print(np.mean(p_labels==yt))
# If everything is fine, you will get the accuracy about 0.725 (training-free)
