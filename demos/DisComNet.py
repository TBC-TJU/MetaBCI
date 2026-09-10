import numpy as np

from metabci.brainda.algorithms.deep_learning.dis_comnet import DisComNet
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_shuffle_indices, match_shuffle_indices)
from scipy import signal

dataset = Wang2016()
events = dataset.events.keys()
freq_list = [dataset.get_freq(event) for event in events]

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 1.14)],  # Tw = 1
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


def get_iir_sos_band(w_pass, w_stop, fs_down):
    '''
    Get second-order sections (like 'ba') of Chebyshev type I filter.
    :param w_pass: list, 2 elements
    :param w_stop: list, 2 elements
    :return: sos_system
        i.e the filter coefficients.
    '''
    if len(w_pass) != 2 or len(w_stop) != 2:
        raise ValueError('w_pass and w_stop must be a list with 2 elements.')

    if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
        raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

    if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
        raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

    wp = [2 * w_pass[0] / fs_down, 2 * w_pass[1] / fs_down]
    ws = [2 * w_stop[0] / fs_down, 2 * w_stop[1] / fs_down]
    gpass = 4
    gstop = 30  # dB

    N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
    sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

    return sos_system


def filtered_data_iir(w_pass_2d, w_stop_2d, data, num_filter, fs_down):
    '''
    filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
    :param w_pass_2d: 2-d, numpy,
        w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
        w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
    :param w_stop_2d: 2-d, numpy,
        w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
        w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
    :param data: 4-d, numpy, from method load_data or resample_data.
        n_chans * n_samples * n_classes * n_trials
    :return: filtered_data: dict,
        {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
        values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
    e.g.
    w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
    w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
    '''
    if w_pass_2d.shape != w_stop_2d.shape:
        raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
    if num_filter > w_pass_2d.shape[1]:
        raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

    # w_pass_2d = np.array([[28, 58], [90, 90]])  # 30, 60
    # w_stop_2d = np.array([[24, 54], [92, 92]])

    sos_system = dict()
    filtered_data = dict()
    for idx_filter in range(num_filter):
        sos_system['filter' + str(idx_filter + 1)] = get_iir_sos_band(
            w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
            w_stop=[w_stop_2d[0, idx_filter],
                    w_stop_2d[1, idx_filter]], fs_down=fs_down)
        filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=2)
        filtered_data['bank' + str(idx_filter + 1)] = filter_data

    return filtered_data


X_raw, y, meta = paradigm.get_data(
    dataset,
    subjects=[23],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# define the filter range
num_filter = 3
w_pass_2d = np.array([[7, 15, 23], [90, 90, 90]])  # 70
w_stop_2d = np.array([[6, 14, 22], [92, 92, 92]])  # 72
filtered_data = filtered_data_iir(w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, data=X_raw, num_filter=num_filter,
                                  fs_down=250)
X = np.zeros((X_raw.shape[0], num_filter, X_raw.shape[1], X_raw.shape[2]))
for idx_filter in range(num_filter):
    X[:, idx_filter, :, :] = filtered_data['bank' + str(idx_filter + 1)]

set_random_seeds(38)
n_splits = 6
# train and validate set will be merged
indices = generate_shuffle_indices(meta, n_splits=n_splits, train_size=4, validate_size=1, test_size=1)  # Nt = 3

accs = []

for k in range(n_splits):
    train_ind, validate_ind, test_ind = match_shuffle_indices(k, meta, indices)
    train_ind = np.concatenate((train_ind, validate_ind))
    X_train, y_train = X[train_ind], y[train_ind]
    discomnet = DisComNet(datalen=X_train.shape[-1], Nk=3, Nb=3)
    accs.append(discomnet.fit(X_train, y_train).score(X[test_ind], y[test_ind]))
    print(accs)

print('Dis-ComNet:  ', np.mean(accs))

# If everything is fine, you will get the accuracy about:
# Dis-ComNet:  0.8958333333
