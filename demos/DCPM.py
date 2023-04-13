import numpy as np
from metabci.brainda.datasets import Xu2018MinaVep
from metabci.brainda.paradigms import aVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    match_char_kfold_indices,
    generate_char_indices
)
from metabci.brainda.algorithms.decomposition.base import (
    TimeDecodeTool
)
from metabci.brainda.algorithms.decomposition import DCPM

dataset = Xu2018MinaVep()

paradigm = aVEP(
    channels=[
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
        'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3',
        'POZ', 'PO4', 'PO6', 'PO8', 'O1',
        'OZ', 'O2'
    ],
)


def raw_hook(raw, caches):
    # do something with raw continuous data
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def trial_hook(raw, caches):
    # do something with trail raw object.
    raw.filter(2, 21, l_trans_bandwidth=1, h_trans_bandwidth=4,
               phase='zero-double')
    caches['trial_stage'] = caches.get('trial_stage', -1) + 1
    return raw, caches


def epochs_hook(epochs, caches):
    # do something with epochs object
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches


def data_hook(X, y, meta, caches):
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches


paradigm.register_raw_hook(raw_hook)
paradigm.register_trial_hook(trial_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

# Time decode tool
TimeDecodeTool = TimeDecodeTool(dataset=dataset)

subject_epoch_acc = []
subject_speller_acc = []

subject = [2]

X, y, meta = paradigm.get_data(
    dataset,
    subjects=subject,
    verbose=False
)

# 3-time leave one out validation
set_random_seeds(38)
k_loo = 3
indices = generate_char_indices(meta, kfold=6)

# classifier
estimator = DCPM(n_components=8)
epoch_accs = []
speller_accs = []
for k in range(k_loo):
    train_ind, validate_ind, test_ind = match_char_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    X_train = np.concatenate([X[i] for i in train_ind])
    y_train = np.concatenate([y[i] for i in train_ind])

    X_test_t = [X[i] for i in test_ind]
    y_test_t = [y[i] for i in test_ind]
    X_test = np.concatenate(X_test_t)
    y_test = np.concatenate(y_test_t)
    label_test = list(meta.event[test_ind])
    model = estimator.fit(X_train, y_train-1)
    p_labels = model.predict(X_test)
    epoch_accs.append(np.mean(p_labels == y_test-1))
    # next evaluate the speller accuracy
    right_count = 0
    test_count = 0
    for test_i in range(len(X_test_t)):
        test_trial = X_test_t[test_i]
        test_feature = estimator.transform(test_trial)
        decode_key = TimeDecodeTool.decode(label_test[test_i], test_feature, fold_num=6)
        if decode_key == label_test[test_i]:
            right_count += 1
    speller_accs.append(right_count/len(X_test_t))

print("Average epoch decode accuracy of subject {} "
      "is: {}".format(subject, np.mean(epoch_accs)))
# If everything is fine, you will get the accuracy about 0.8763.

print("Average speller decode accuracy of subject {} "
      "is: {}".format(subject, np.mean(speller_accs)))
# If everything is fine, you will get the accuracy about 0.8958.
