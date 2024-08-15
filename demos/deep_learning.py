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
from metabci.brainda.datasets.tsinghua import Wang2016  # 使用 Tsinghua 数据集
from metabci.brainda.paradigms import SSVEP  # 使用 SSVEP 范式
from metabci.brainda.algorithms.deep_learning.cnn_gru_attn import CNN_GRU_Attn

dataset = Wang2016()  # 使用 Wang2016 数据集
paradigm = SSVEP(
    channels=None,
    events=['8', '9'],  
    intervals=[[0.5, 2.5]],  
    srate=None
)

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],  
    return_concat=True,
    n_jobs=None,
    verbose=False)

set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# 使用CNN_GRU_Attn模型
estimator = CNN_GRU_Attn(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=2)

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels == y[test_ind]))

print(np.mean(accs))

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