# decode Cattan_P300 dataset
import numpy as np
from metabci.brainda.datasets import Cattan_P300
from metabci.brainda.paradigms import P300
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_char_indices,
    match_char_kfold_indices
)
from metabci.brainda.algorithms.decomposition.base import (
    TimeDecodeTool
)
from metabci.brainda.algorithms.decomposition import DCPM


dataset = Cattan_P300()

paradigm = P300(
    channels=[
        'FP1', 'FP2', 'FC5', 'FZ', 'FC6', 'T7', 'CZ',
        'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2'
    ]
)


def raw_hook(raw, caches):
    # do something with raw continuous data
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def trial_hook(raw, caches):
    # do something with trail raw object.
    raw.filter(1, 20, l_trans_bandwidth=1, h_trans_bandwidth=4,
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
TimeDecodeTool1 = TimeDecodeTool(dataset=dataset)

subject_epoch_acc = []
subject_speller_acc = []

subject = [2]

X, y, meta = paradigm.get_data(
    dataset,
    subjects=subject,
    verbose=False
)

# 12-time leave one out validation
set_random_seeds(38)
k_loo = 12
indices = generate_char_indices(meta, kfold=k_loo)

# classifier
estimator = DCPM(n_components=8)
epoch_accs = []
speller_accs = []
result = []
for k in range(k_loo):
    train_id, val_id, test_id \
        = match_char_kfold_indices(k, meta, indices)
    train_ind = np.concatenate((train_id, val_id))
    test_ind = test_id
    X_train_t = [X[i] for i in train_ind]
    y_train_t = [y[i] for i in train_ind]
    Key = meta.event[train_ind]
    y_train_tar = TimeDecodeTool1.target_calibrate(y_train_t, Key)

    X_train = np.concatenate(X_train_t)
    y_train = np.concatenate(y_train_tar)
    model = estimator.fit(X_train, y_train-1)

    X_test_t = [X[i] for i in test_id]
    y_test_t = [y[i] for i in test_id]

    X_test_sort, y_test_sort = TimeDecodeTool1.epoch_sort(X_test_t, y_test_t)
    label_test = list(meta.event[test_ind])
    y_test_tar = TimeDecodeTool1.target_calibrate(y_test_sort, Key)
    right_count = 0
    for test_i in range(len(X_test_sort)):
        X_test = X_test_sort[test_i]
        y_test = y_test_tar[test_i]
        test_feature = estimator.transform(X_test)
        decode_key = TimeDecodeTool1.decode(
            label_test[test_i],
            test_feature,
            fold_num=5,
            paradigm=dataset.paradigm)
        result.append(decode_key)
        if decode_key == label_test[test_i]:
            right_count += 1
        speller_accs.append(right_count / len(X_test_t))

print("Average speller decode accuracy of subject {} "
      "is: {}".format(subject, np.mean(speller_accs)))
# If everything is fine, you will get the accuracy about 0.8333.
