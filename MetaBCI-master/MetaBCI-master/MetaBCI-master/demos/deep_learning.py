import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from metabci.brainda.algorithms.deep_learning import ConvCA, EEGNet
from metabci.brainda.algorithms.deep_learning.guney_net import GuneyNet
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from metabci.brainda.algorithms.decomposition import CSP
from metabci.brainda.algorithms.deep_learning.shallownet import ShallowNet
from metabci.brainda.algorithms.deep_learning.deepnet import Deep4Net
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery

dataset = AlexMI()  # declare the dataset
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=None,
    srate=None
)  # declare the paradigm, use recommended Options

# X,y are numpy array and meta is pandas dataFrame
X, y, meta = paradigm.get_data(
    dataset,
    subjects=[8],
    return_concat=True,
    n_jobs=None,
    verbose=False)

set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# assume we have a X with size [batch size, number of channels, number of sample points]
# for shallownet/deepnet/eegnet, you can write like this: estimator = EEGNet(X.shape[1], X.shape[2], 2)

# for GuneyNet, you will have a X with size
# [batch size, number of channels, number of sample points, number of sub_bands]
# and you need to transpose it or in other words switch the dimension of X
# to make X size be like [batch size, number of sub_bands, number of channels, number of sample points]
# and initialize guney like this: estimator = GuneyNet(X.shape[2], X.shape[3], 2, 3)

# for convCA, you will also need a T(reference signal), you can initialize network like shallownet by
# estimator = ConvCA(X.shape[1], X.shape[2], 2),
# but you need to wrap X and T in a dict like this {'X': X, 'T', T} to train the network
# like this:
# dict = {'X': train_X, 'T', T}
# estimator.fit(dict, train_y).
#
# the size of X and T need to be
# X: [batch size, number of channels, number of sample points]
# T: [batch size, number of channels, number of classes, number of sample points]


estimator = ShallowNet(X.shape[1], X.shape[2], 2)

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels==y[test_ind]))
print(np.mean(accs))

