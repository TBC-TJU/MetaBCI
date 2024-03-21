import numpy as np

from metabci.brainda.algorithms.deep_learning import ShallowNet
from metabci.brainda.algorithms.deep_learning.pretraining import PreTraing
from metabci.brainda.datasets import BNCI2014001, Schirrmeister2017
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)

# path to save the pre-train model weight
save_path = './model.pkl'

# use Highgamma dataset as the source dataset
source_dataset = Schirrmeister2017()  # declare the dataset
source_paradigm = MotorImagery(
    channels=['C3', 'CZ', 'C4'],
    events=['left_hand', 'right_hand', 'feet', 'rest'],
    intervals=[(-0.5, 4)],
    srate=250
)
# use the BCI4 2a dataset as the target dataset
target_dataset = BNCI2014001()
target_paradigm = MotorImagery(
    channels=['C3', 'CZ', 'C4'],
    events=['left_hand', 'right_hand', 'feet', 'tongue'],
    intervals=[(-0.5, 4)],
    srate=250
)
# note that source and target dataset should have the same channels, intervals, and srate

# get all data from the source dataset
num_subject = len(source_dataset.subjects)
all_subject_x = []
all_subject_y = []
for s_id in range(num_subject):
    X, y, _ = source_paradigm.get_data(
        source_dataset,
        subjects=[s_id+1],
        return_concat=True,
        n_jobs=None,
        verbose=False)
    all_subject_x.append(X)
    all_subject_y.append(y)
all_subject_x = np.concatenate(all_subject_x)
all_subject_y = np.concatenate(all_subject_y)

# get single subject data in target dataset, then fine-tuning&testing
X, y, meta = target_paradigm.get_data(
        target_dataset,
        subjects=[9],
        return_concat=True,
        n_jobs=None,
        verbose=False)

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

# initializing model for pre-training and pre-train method
target_n_class = 4  # 'left_hand', 'right_hand', 'feet', 'rest'
source_n_class = 4  # 'left_hand', 'right_hand', 'feet', 'tongue'

# size_before_classification is the feature size
# of the last feature extraction layer(layer before the linear layer)'s output
# We can get the value of size_before_classification
# by looking up input size of the linear layer in deep learning models
# in this demo, you can look up the size in estimator.module.fc_layer.weight
size_before_classification = 2760

# pre_training, you can replace the source_estimator with other deep learning model
# but the applied model should have a class method named 'cal_backbone'
# 'cal_backbone' process the input x just like the 'forward' function do
# except it do not use linear layer to classify the extracted feature by the frontal layers
# 'cal_backbone' has implemented in shallownet, deepnet, and eegnet
# please see these net for examples
source_estimator = ShallowNet(all_subject_x.shape[1], all_subject_x.shape[2], source_n_class)
tflm = PreTraing(target_n_class, size_before_classification)
tflm.pretraining(source_estimator, save_path, all_subject_x, all_subject_y)

# compare accuracy between model with/without pretrain
accs_wop = []
accs_wp = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    train_ind = np.concatenate((train_ind, validate_ind))
    # fine-tuning model with pre-train
    estimator = tflm.finetuning(source_estimator, save_path, X[train_ind], y[train_ind])
    p_labels = estimator.predict(X[test_ind])
    accs_wp.append(np.mean(p_labels==y[test_ind]))

    # common training and testing
    target_estimator = ShallowNet(X.shape[1], X.shape[2], target_n_class)
    p_labels = target_estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs_wop.append(np.mean(p_labels == y[test_ind]))

print("Accuracy without pretraining:", np.mean(accs_wop))
print("Accuracy with pretraining", np.mean(accs_wp))
