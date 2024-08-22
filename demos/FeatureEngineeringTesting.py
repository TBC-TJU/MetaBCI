import numpy as np
import csv
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import dump,load
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import random
import torch.nn.functional as F
from thop import profile
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.feature_engineering import emg_features 
from metabci.brainda.algorithms.deep_learning.Fusion import Late_Fusion_Attention,Unimodal,Late_Fusion
from metabci.brainda.algorithms.decomposition import SKLDA
from metabci.brainda.algorithms.decomposition import STDA
from skorch.callbacks import Checkpoint

from metabci.brainda.algorithms.feature_engineering.ExtractingFeatures import Extracting_Feature


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
setup_seed(20)




fs_eeg = 1000 # 采样频率
lowcut = 0.5 # 低频截止频率
highcut = 150 # 高频截止频率
nyquist = 0.5 * fs_eeg
low = lowcut / nyquist
high = highcut / nyquist
order = 4
b11_low, a11_low     = signal.butter(order, [0.05/nyquist, 0.5/nyquist], btype='band', analog=False)
b11_delta, a11_delta = signal.butter(order, [0.5/ nyquist, 4/ nyquist ], btype='band', analog=False)
b11_theta, a11_theta = signal.butter(order, [4 / nyquist,  8/ nyquist ], btype='band', analog=False)
b11_alpha, a11_alpha = signal.butter(order, [8/ nyquist,   13/ nyquist], btype='band', analog=False)
b11_belta, a11_belta = signal.butter(order, [13/ nyquist,  30/ nyquist], btype='band', analog=False)
b11_gamma, a11_gamma = signal.butter(order, [30/ nyquist, 100/ nyquist], btype='band', analog=False)
b11_high, a11_high   = signal.butter(order, [100/ nyquist, 150/nyquist], btype='band', analog=False)

# 设计工频陷波滤波器
notch_freq = 50 # 工频频率为60Hz
Q = 30 # 品质因数
b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs_eeg)

# emg设计带通滤波器
fs_emg = 1000 # 采样频率
lowcut = 10  # 低频截止频率
highcut = 300 # 高频截止频率
nyquist = 0.5 * fs_emg
low = lowcut / nyquist
high = highcut / nyquist
order = 4
b12, a12 = signal.butter(order, [low, high], btype='band', analog=False)

filtering_type = 'bandpass'

def segmentation(Rawdata,Label_1,length_eeg,length_emg,step):
    if length_eeg==length_emg:
        window_length = length_eeg  #ms eeg segmentation
    else:
        window_length = length_emg  #ms emg segmentation

    step = step #ms
    data = []
    label_1 = []
    #label_2 = []
    end_point = 0

    print("Label.shape:",np.array(Label_1).shape,np.unique(np.array(Label_1), return_counts=True))
    #print("Label_2.shape:",np.array(Label_2).shape,np.unique(np.array(Label_2), return_counts=True))

    for i,rawdata in enumerate(Rawdata):
        while (end_point-1<rawdata.shape[1]):
            if end_point==0:
                end_point = length_eeg
                data.append(rawdata[:,end_point-window_length:end_point])
                end_point = end_point + step
                label_1.append(Label_1[i])  
                #label_2.append(Label_2[i])  
            else:
                data.append(rawdata[:,end_point-window_length:end_point])
                end_point = end_point + step
                label_1.append(Label_1[i])  
                #label_2.append(Label_2[i])
        end_point = 0
    print("number of samples:",len(data))
    # print("label.shape:",np.array(label).shape,np.unique(np.array(label), return_counts=True))
    return data,label_1

def prepare_dataset(length_eeg,step,EEG_All,Label_1_All):
    train_eeg,train_label_1 = segmentation(EEG_All[0:20*3],Label_1_All[0:20*3],length_eeg,length_eeg,step)#前4轮数据做训练数据
    #train_emg,train_label_1,train_label_2 = segmentation(EMG_All[0:20*3],Label_1_All[0:20*3],Label_2_All[0:20*3],length_eeg,length_emg,step)

    test_eeg,test_label_1 = segmentation(EEG_All[20*3:],Label_1_All[20*3:],length_eeg,length_eeg,step)
    #test_emg,test_label_1,test_label_2 = segmentation(EMG_All[20*3:],Label_1_All[20*3:],Label_2_All[20*3:],length_eeg,length_emg,step)
    
    train_data = [train_eeg]
    test_data = [test_eeg]
    # print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))
    # print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))

    return train_data,train_label_1,test_data,test_label_1

##调用新增的特征提取功能
FeatureEngineering = Extracting_Feature(threshold  = 1e-5, EPS = 1e-5, fs=100, type='MEDIAN_POWER')
def extract_emg_feature(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(FeatureEngineering.predict(x[i,:],feature_name))
    res =np.vstack(res)
    return res

def Predcit(length_eeg,step,X,y, train):
    if train == True:
        Train_data,train_label,Test_data,test_label = prepare_dataset(length_eeg,step,X,y)
        data = np.array(Train_data[0])
        label = train_label
    else:
        Train_data,train_label,Test_data,test_label = prepare_dataset(length_eeg,step,X,y)
        data = np.array(Test_data[0])
        label = test_label
    # eeg_feature = np.array(train_data[0])#np.fft.fft(train_data[0],axis=1).real#
    # # eeg_feature1 = np.expand_dims(signal.filtfilt(b11_low, a11_low, eeg_feature,axis=1),axis = 1)
    # eeg_feature1 = np.expand_dims(eeg_feature,axis = 1) #- eeg_feature1
    # eeg_feature2 = np.expand_dims(signal.filtfilt(b11_delta, a11_delta, eeg_feature,axis=1),axis = 1)
    # eeg_feature3 = np.expand_dims(signal.filtfilt(b11_theta, a11_theta, eeg_feature,axis=1),axis = 1)
    # eeg_feature4 = np.expand_dims(signal.filtfilt(b11_alpha, a11_alpha, eeg_feature,axis=1),axis = 1)
    # eeg_feature5 = np.expand_dims(signal.filtfilt(b11_belta, a11_belta, eeg_feature,axis=1),axis = 1)
    # eeg_feature6 = np.expand_dims(signal.filtfilt(b11_gamma, a11_gamma, eeg_feature,axis=1),axis = 1)
    # eeg_feature7 = np.expand_dims(signal.filtfilt(b11_high, a11_high, eeg_feature,axis=1),axis = 1)

    # eeg_feature = np.concatenate((eeg_feature1,eeg_feature2,eeg_feature3,\
    #                               eeg_feature4,eeg_feature5,eeg_feature6,\
    #                               eeg_feature7),axis=1)#
    
    feature_list = ['ssc','wl','mean','rms','log','mnf_MEDIAN_POWER']#]#,'mav','var']#,'psr']#,'arc'
    emg_feature = []
    for feature_name in feature_list:
        #data = np.array(train_data[0])#[:,:,::10]
        data1 = data[:,:,:data.shape[2]//2]
        data2 = data[:,:,data.shape[2]//2:]
        feature = np.expand_dims([np.transpose(extract_emg_feature(seg, feature_name)) for seg in data],axis = 1)
        feature1 = np.expand_dims([np.transpose(extract_emg_feature(seg, feature_name)) for seg in data1],axis = 1)
        feature2 = np.expand_dims([np.transpose(extract_emg_feature(seg, feature_name)) for seg in data2],axis = 1)
        feature = np.concatenate((feature,feature1,feature2),axis=1)
        if len(emg_feature)==0:
            emg_feature =  feature
        else:
            emg_feature = np.concatenate((emg_feature,feature),axis=2)
    #     emg_feature.append(feature)
    # emg_feature = emg_feature.reshape(10,-1)
    print("Feature Engineering is done!")
    return [emg_feature],label

dataset = Wang2016()
paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],
    srate=250
)
X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

length_eeg,length_emg,step = 500,200,100
#Train_data,train_label,Test_data,test_label = prepare_dataset(length_eeg,step,X,y)
Train_data,train_label = Predcit(length_eeg,step,X,y,train=True)
print("x.shape:",X.shape,y,y.shape)
    ##np.hstack((np.array(Train_data[0]).reshape(np.array(Train_data[0]).shape[0],-1),np.array(Train_data[1]).reshape(np.array(Train_data[0]).shape[0],-1)))
      
    # train_data = np.hstack((train_data,np.fft.fft(train_data,axis=2).real))
    # train_data = np.fft.fft(train_data,axis=2).imag
    # train_data_eeg = (train_data-train_data.mean(axis=0, keepdims=True))/train_data.std(axis=0, keepdims=True)
    # train_data = np.array(Train_data[1])#np.hstack((list(train_data[0]),list(train_data[1])))
    # train_data_emg = (train_data-train_data.mean(axis=0, keepdims=True))/train_data.std(axis=0, keepdims=True)
    # train_data = train_data_eeg#np.hstack((train_data_eeg,train_data_emg))#
print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))
#print("train_label_2.shape:",np.array(train_label_2).shape,np.unique(np.array(train_label_2), return_counts=True))

Test_data,test_label = Predcit(length_eeg,step,X,y,train=False)
print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))
#print("test_label_2.shape:",np.array(test_label_2).shape,np.unique(np.array(test_label_2), return_counts=True))


##USTChopin的算法Late_Fusion识别metabci中的数据
setup_seed(9999)
X1 = X[:,:4,:].reshape(X.shape[0],1,125,4)
X2 = X[:,4:,:].reshape(X.shape[0],1,125,X.shape[1]-4)
print("x.shape:",X1.shape,y.shape)
model = Late_Fusion(dropout=0.15,input_dim1= X1.shape[2]*X1.shape[3],
                    channel_eeg=X1.shape[1],input_dim2=X2.shape[2]*X2.shape[3],
                    channel_emg=X2.shape[1],classes=40)
criterion1 = nn.CrossEntropyLoss()
# 数据加载器
train_data = [torch.tensor(X1,dtype=torch.float32),
              torch.tensor(X2,dtype=torch.float32)]
label = torch.tensor(np.array(y[:]),dtype=torch.long)
model.fit([train_data[0], train_data[1]], label)
# test_data1 = np.expand_dims(X[1::2,4:,:], axis=1) 
# test_data2 = np.expand_dims(X[1::2,4:,:], axis=1) 
test_data1 = X[:,:4,:].reshape(X.shape[0],1,125,4)
test_data2 = X[:,4:,:].reshape(X.shape[0],1,125,X.shape[1]-4)
testdata = Test_data
test_data = [torch.tensor(test_data1,dtype=torch.float),
             torch.tensor(test_data2,dtype=torch.float)]
test_label = np.array(y)
y_pred = model.predict([test_data[0], test_data[1]])
y_pred = np.array(y_pred)
accuracy_loaded = accuracy_score(test_label, y_pred)
print(f"Accuracy with loaded model: {accuracy_loaded:.4f}")    





