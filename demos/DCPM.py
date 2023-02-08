import numpy as np
from metabci.brainda.datasets import Xu2018MinaVep
from metabci.brainda.paradigms import aVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices,
    match_loo_indices_dict
)
from metabci.brainda.utils.time_encode_tool import concat_trials, TimeDecodeTool
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
indices = generate_loo_indices(meta)

# classifier
estimator = DCPM(n_components=8)
epoch_accs = []
speller_accs = []
for k in range(k_loo):
    X_train, y_train, X_dev, y_dev, X_test, y_test \
        = match_loo_indices_dict(X, y, meta, indices, k)
    train_X_t, train_y_t = concat_trials(X_train, y_train)
    dev_X_t, dev_y_t = concat_trials(X_dev, y_dev)
    train_X = np.concatenate((train_X_t, dev_X_t), axis=0)
    train_y = np.concatenate((train_y_t, dev_y_t), axis=0)
    model = estimator.fit(train_X, train_y-1)
    # first evaluate the epochs decode accuracy
    test_X_c, test_y_c = concat_trials(X_test, y_test)
    p_labels = model.predict(test_X_c)
    epoch_accs.append(np.mean(p_labels == test_y_c-1))
    # next evaluate the speller accuracy
    right_count = 0
    test_count = 0
    for key, value in X_test.items():
        for epoch in value:
            test_count += 1
            epoch_feature = estimator.transform(epoch)
            decode_key = TimeDecodeTool.decode(key, epoch_feature)
            if decode_key == key:
                right_count += 1
    speller_accs.append(right_count/test_count)

print("Average epoch decode accuracy of subject {} "
      "is: {}".format(subject, np.mean(epoch_accs)))
# If everything is fine, you will get the accuracy about 0.8861.

print("Average speller decode accuracy of subject {} "
      "is: {}".format(subject, np.mean(speller_accs)))
# If everything is fine, you will get the accuracy about 0.9063.
