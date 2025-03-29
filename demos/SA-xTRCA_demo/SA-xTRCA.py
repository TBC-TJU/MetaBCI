import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
METABCI_PATH = os.path.join(PROJECT_ROOT, 'metabci')
if METABCI_PATH not in sys.path:
    sys.path.insert(0, METABCI_PATH)

# 加载数据路径
DATA_PATH = os.path.join(CURRENT_DIR, 'data')
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.decomposition import SAxTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from scipy.io import loadmat 
from sklearn.model_selection import KFold 


for sub_i in range(10):
    # load data
    file_path = DATA_PATH + '\\sub' + str(sub_i+1) + '\\1.mat'
    mat_file = loadmat(file_path)
    data = mat_file['data']
    data = np.transpose(data,(3,0,1,2))
    N_trial = data.shape[0]
    N_chan = data.shape[1]
    N_sample = data.shape[2]
    N_type = data.shape[3]
    # Initial latency vector for each trial, representing the assumed starting point (in samples) of each trial.
    t0 = [75 for i in range(N_trial)]
    # 采样率
    fs = 250
    # The number of sampling points per trial
    tau = 500
    # The search range (in samples) for the sliding window when estimating latency.
    tsearch0 = 50
    # The annealing coefficient used in the simulated annealing (SA) process to control temperature reduction.
    r = 0.9
    # The maximum number of consecutive iterations allowed without improvement before forced termination 
    # of the optimization process.
    Q = 10

    estimator = SAxTRCA(n_components=1, t0=t0, tau=tau, tsearch0=tsearch0, r=r, Q=Q)
    
    X_align = np.zeros((9,21,500,10))
    X_all = np.concatenate([data[i,:,:,:] for i in range(N_trial)],axis=-1)
    X_all = np.transpose(X_all,(2,0,1))
    y = np.squeeze(np.concatenate(np.tile(np.arange(0,N_type),(N_trial,1)) ,axis=0))
    N_fold = 10
    kf = KFold(N_fold)
    accs = []

    for train_ind, test_ind in kf.split(X_all):
        estimator.fit(X_all[train_ind], y[train_ind])
        p_labels = estimator.predict(X_all[test_ind])
        accs.append(np.mean(p_labels==y[test_ind]))
    print('Sub' + str(sub_i+1) + ': accuracy is ',np.mean(accs))
    