# import pyEDFlib
import numpy as np
import csv
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import numpy as np
from scipy import signal
from joblib import dump,load
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import random
import torch.nn.functional as F
from thop import profile

# from metabci.brainda.algorithms.feature_engineering import emg_features
from metabci.brainda.algorithms.deep_learning.Fusion import Late_Fusion_Attention,Unimodal,Late_Fusion
from metabci.brainda.algorithms.decomposition import SKLDA
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

# eeg设计带通滤波器
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



def get_rawdata(experiment_data):
    EEG_All = []
    EMG_All = []
    Label_1_All = []
    Label_2_All = []
    # 打印实验数据
    for data in experiment_data:
        raw = data['data']
        raw = signal.filtfilt(b_notch, a_notch,raw,axis=1)
        list_eeg = range(0,30)#(7,20)#(2,25)#(7,8,9,10,12,13,14,)#
        list_emg = (35,40,39,33,36,51,56,53,46,48)#,)#
        if raw[list_eeg,:].shape[1]>800:
            eeg_filtered = raw[list_eeg,:] #signal.filtfilt(b11, a11,raw[list_eeg,:],axis=1)
            eeg_filtered = np.transpose(eeg_filtered,(1,0)) - np.mean(eeg_filtered[:,:1000],1).reshape(1,-1)
            EEG_All.append(np.transpose(eeg_filtered,(1,0))[:,1000:])
            # EEG_All.append(eeg_filtered[:,2000:])

            emg_filtered = signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)
            TEMP = np.mean(emg_filtered[:,:1000],1).reshape(1,-1)
            emg_filtered = np.transpose(emg_filtered,(1,0)) - TEMP
            EMG_All.append(np.abs(np.transpose(emg_filtered,(1,0))[:,1000:]))
            # EMG_All.append(emg_filtered[:,2000:])
        else:#if raw[list_eeg,:].shape[1]==800:
            EEG_All.append(raw[list_eeg,:])
            # EEG_All.append(signal.filtfilt(b11, a11, raw[list_eeg,:],axis=1))
            EMG_All.append(np.abs(signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)))
            # EMG_All.append(raw[list_emg,:])
 
        Label_1_All.append(np.array(data['marker1']))
        Label_2_All.append(np.array(data['marker2']))
    print("data['data'].shape:",data['data'].shape)
    print("EEG_All.shape:",len(EEG_All))
    print("EMG_All.shape:",len(EMG_All))
    print("Label_1_All.shape:",np.array(Label_1_All).shape,np.unique(np.array(Label_1_All), return_counts=True))
    print("Label_2_All.shape:",np.array(Label_2_All).shape,np.unique(np.array(Label_2_All), return_counts=True))
    return EEG_All,EMG_All,Label_1_All,Label_2_All


def get_rawdata_atten(Atten_train, Atten_test):
    EEG_All = []
    # 打印实验数据
    for data in Atten_train:
        raw = data['data']
        raw = signal.filtfilt(b_notch, a_notch, raw, axis=1)
        list_eeg = range(0, 30)
        eeg_filtered = raw[list_eeg, :].T
        EEG_All.append(eeg_filtered)
    for data in Atten_test:
        raw = data['data']
        raw = signal.filtfilt(b_notch, a_notch, raw, axis=1)
        list_eeg = range(0, 30)
        eeg_filtered = raw[list_eeg, :].T
        EEG_All.append(eeg_filtered)
    EEG_All = np.array(EEG_All)
    print("EEG_All.shape:", EEG_All.shape)

    return EEG_All

def segmentation(Rawdata,Label_1,Label_2,length_eeg,length_emg,step):
    if length_eeg==length_emg:
        window_length = length_eeg  #ms eeg segmentation
    else:
        window_length = length_emg  #ms emg segmentation

    step = step #ms
    data = []
    label_1 = []
    label_2 = []
    end_point = 0

    print("Label.shape:",np.array(Label_1).shape,np.unique(np.array(Label_1), return_counts=True))
    print("Label_2.shape:",np.array(Label_2).shape,np.unique(np.array(Label_2), return_counts=True))

    for i,rawdata in enumerate(Rawdata):
        while (end_point-1<rawdata.shape[1]):
            if end_point==0:
                end_point = length_eeg
                data.append(rawdata[:,end_point-window_length:end_point])
                end_point = end_point + step
                label_1.append(Label_1[i])  
                label_2.append(Label_2[i])  
            else:
                data.append(rawdata[:,end_point-window_length:end_point])
                end_point = end_point + step
                label_1.append(Label_1[i])  
                label_2.append(Label_2[i])
        end_point = 0
    print("number of samples:",len(data))
    # print("label.shape:",np.array(label).shape,np.unique(np.array(label), return_counts=True))
    return data,label_1,label_2

def prepare_dataset(length_eeg,length_emg,step,EEG_All,EMG_All,Label_1_All,Label_2_All):
    train_eeg,train_label_1,train_label_2 = segmentation(EEG_All[0:20*3],Label_1_All[0:20*3],Label_2_All[0:20*3],length_eeg,length_eeg,step)#前4轮数据做训练数据
    train_emg,train_label_1,train_label_2 = segmentation(EMG_All[0:20*3],Label_1_All[0:20*3],Label_2_All[0:20*3],length_eeg,length_emg,step)

    test_eeg,test_label_1,test_label_2 = segmentation(EEG_All[20*3:],Label_1_All[20*3:],Label_2_All[20*3:],length_eeg,length_eeg,step)
    test_emg,test_label_1,test_label_2 = segmentation(EMG_All[20*3:],Label_1_All[20*3:],Label_2_All[20*3:],length_eeg,length_emg,step)
    
    train_data = [train_eeg,train_emg]
    test_data = [test_eeg,test_emg]
    print('eeg', np.array(train_eeg).shape, np.array(test_eeg).shape)
    # print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))
    # print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))

    return train_data,train_label_1,train_label_2,test_data,test_label_1,test_label_2


def prepare_dataset_atten(win_len, step, Atten_EEG):
    train_eeg, test_eeg = Atten_EEG[0:-2], Atten_EEG[-2:]
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for n in range(len(train_eeg)):
        i = 0
        while step * i + win_len <= train_eeg.shape[1]:
            data = train_eeg[n, step * i : step * i + win_len].T
            train_data.append(data)
            # 专注标签1，不专注标签0
            train_label.append((n+1) % 2)
            i += 1
    for n in range(len(test_eeg)):
        i = 0
        while step * i + win_len <= test_eeg.shape[1]:
            data = test_eeg[n, step * i : step * i + win_len].T
            test_data.append(data)
            # 专注标签1，不专注标签0
            test_label.append((n+1) % 2)
            i += 1
    train_data, train_label, test_data, test_label = np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

    return train_data, train_label, test_data, test_label

#调用新增的特征提取功能
FeatureEngineering = Extracting_Feature(threshold  = 1e-5, EPS = 1e-5, fs=100, type='MEDIAN_POWER')
def extract_emg_feature(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(FeatureEngineering.predict(x[i,:],feature_name))
    res =np.vstack(res)
    return res



def extract_emg_feature_old(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(eval(str(func))(x[i,:]))
    res =np.vstack(res)
    return res

def Feature_Engineering(train_data,train_label_1,train_label_2):
    eeg_feature = np.array(train_data[0])#np.fft.fft(train_data[0],axis=1).real#
    # eeg_feature1 = np.expand_dims(signal.filtfilt(b11_low, a11_low, eeg_feature,axis=1),axis = 1)
    eeg_feature1 = np.expand_dims(eeg_feature,axis = 1) #- eeg_feature1
    eeg_feature2 = np.expand_dims(signal.filtfilt(b11_delta, a11_delta, eeg_feature,axis=1),axis = 1)
    eeg_feature3 = np.expand_dims(signal.filtfilt(b11_theta, a11_theta, eeg_feature,axis=1),axis = 1)
    eeg_feature4 = np.expand_dims(signal.filtfilt(b11_alpha, a11_alpha, eeg_feature,axis=1),axis = 1)
    eeg_feature5 = np.expand_dims(signal.filtfilt(b11_belta, a11_belta, eeg_feature,axis=1),axis = 1)
    eeg_feature6 = np.expand_dims(signal.filtfilt(b11_gamma, a11_gamma, eeg_feature,axis=1),axis = 1)
    eeg_feature7 = np.expand_dims(signal.filtfilt(b11_high, a11_high, eeg_feature,axis=1),axis = 1)

    eeg_feature = np.concatenate((eeg_feature1,eeg_feature2,eeg_feature3,\
                                  eeg_feature4,eeg_feature5,eeg_feature6,\
                                  eeg_feature7),axis=1)#
    feature_list = ['ssc','wl','mean','rms','log','mnf_MEDIAN_POWER']#]#,'mav','var']#,'psr']#,'arc'
    emg_feature = []
    for feature_name in feature_list:
        data = np.array(train_data[1])#[:,:,::10]
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
    return [eeg_feature,emg_feature],train_label_1,train_label_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trian_Fusion(model,learning_rate,train_loader,num_epochs,criterion):#,criterion2):
    # 定义优化器
    for i,lr in enumerate(learning_rate):
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        total_step = len(train_loader)
        for epoch in range(num_epochs[i]):
            correct = 0
            correct_2 = 0
            total = 0
            total_2 = 0
            Outcome = []
            Outcome_attention = []
            Label = []
            Label_2 = []
            for i, (images1, images2, labels, labels_2) in enumerate(train_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)
                labels_2 = labels_2.to(device)

                # 前向传播
                outputs,f_attention,a5 = model(images1,images2)
                # x5,a5 = F.normalize(x5, dim=1),F.normalize(a5, dim=1)
                loss = criterion(outputs, labels)+criterion(f_attention, labels_2)
                    # + kd_loss(x5, a5, 1)\
                    # + criterion2(torch.cat([x5.unsqueeze(1),a5.unsqueeze(1)],1),labels)\

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                _, predicted_attention = torch.max(f_attention.data, 1)
                Outcome.append(predicted)
                Outcome_attention.append(predicted_attention)
                Label.append(labels)
                Label_2.append(labels_2)
                total += labels.size(0)
                total_2 += labels_2.size(0)
                correct += (predicted == labels).sum().item()
                correct_2 += (predicted_attention == labels_2).sum().item()

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Acc: {:.4f} Acc_Attention: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),100 * correct / total,100 * correct_2 / total_2))
        print('Accuracy of the model on the train images: {} % {} %'.format(100 * correct / total,100 * correct_2 / total_2))
    return model,Outcome,Outcome_attention,Label,Label_2

def train_PR_model(train_data,train_label_1,train_label_2,Exp_date,type):
    data = train_data#np.array(train_data[0])#np.hstack((list(train_data[0]),list(train_data[1])))
    label = train_label_1
    label_2 = train_label_2
    if type=="SKLDA":
        model = SKLDA()
        model.fit(np.reshape(data,(data.shape[0],-1)), label)
        return model

    elif type=="LDA":
        model = LDA()
        model.fit(np.reshape(data,(data.shape[0],-1)), label)
        model_2 = LDA()
        model_2.fit(np.reshape(data,(data.shape[0],-1)), label_2)
        return model, model_2

    elif type=="SVM":
        model = SVC(kernel='rbf')  # 可以选择不同的核函数，如'rbf', 'poly','linear'等
        model.fit(np.reshape(data,(data.shape[0],-1)), label)
        model_2 = SVC(kernel='rbf')  # 可以选择不同的核函数，如'rbf', 'poly','linear'等
        model_2.fit(np.reshape(data,(data.shape[0],-1)), label_2)
        return model, model_2

    elif type=="RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(np.reshape(data,(data.shape[0],-1)), label)
        model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_2.fit(np.reshape(data,(data.shape[0],-1)), label_2)
        return model, model_2


    # elif type=="MLP":
    #     setup_seed(9999)
    #     # 实例化模型
    #     model = MLP_Test(dropout=0.05,input_size=data.shape[2]*data.shape[3],
    #                      hiden_size=300,channel=data.shape[1],classes=6).to(device)
    #     img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
    #     img = img.to("cuda")
    #     flops, params = profile(model, inputs=(img,))
    #     print("flops:",flops)
    #     print("params:",params)
    #     print("MLP params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位，

    #     criterion1 = nn.CrossEntropyLoss()
    #     # 数据加载器
    #     train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
    #                                    torch.tensor(np.array(label),dtype=torch.long))
    #     train_loader1 =  DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
    #     model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
    #                                                     train_loader=train_loader1,
    #                                                     num_epochs=[20,20],
    #                                                     criterion=criterion1)
    #     return model

    # elif type == "Unimodal":
    #     setup_seed(9999)
    #     # 实例化模型
    #     model = Unimodal(dropout=0.05,input_dim=data.shape[2]*data.shape[3],
    #                      channel=data.shape[1],classes=6).to(device)
    #     img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
    #     img = img.to("cuda")
    #     flops, params = profile(model, inputs=(img,))
    #     print("flops:",flops)
    #     print("params:",params)
    #     print("Unimodal params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位，
    #
    #     criterion1 = nn.CrossEntropyLoss()
    #
    #     # 数据加载器
    #     train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
    #                                    torch.tensor(np.array(label),dtype=torch.long))
    #     train_loader1 = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
    #     model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
    #                                                     train_loader=train_loader1,
    #                                                     num_epochs=[20,20],
    #                                                     criterion=criterion1)
    #     return model


    elif type == "Late_Fusion":
        setup_seed(9999)
        # 实例化模型dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes
        model = Late_Fusion(dropout=0.15,input_dim1=data[0].shape[2]*data[0].shape[3],
                         channel_eeg=data[0].shape[1],input_dim2=data[1].shape[2]*data[1].shape[3],
                         channel_emg=data[1].shape[1],classes=6)
        img = torch.randn(2,data[0].shape[1],data[0].shape[2],data[0].shape[3])
        img1 = torch.randn(2,data[1].shape[1],data[1].shape[2],data[1].shape[3])
        print("img,img1:",img.shape,img1.shape)
        img1 = img1.to("cpu")
        img = img.to("cpu")
        # flops, params = profile(model, inputs=(img,img1,))
        # print("flops:",flops)
        # print("params:",params)
        # print("Late_Fusion params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位，

        criterion1 = nn.CrossEntropyLoss()

        # 数据加载器
        train_data = [torch.tensor(data[0],dtype=torch.float32),
                                      torch.tensor(data[1],dtype=torch.float32)]

        label = torch.tensor(np.array(label),dtype=torch.long)
        label_2 = torch.tensor(np.array(label_2),dtype=torch.long)
        model.fit([train_data[0], train_data[1]], label)
        # model,Train_Outcome,Train_Outcome_attention,Train_Label_1,Train_Label_2 = trian_Fusion(model,learning_rate=[0.01,0.001],
        #                                                 train_loader=train_loader1,
        #                                                 num_epochs=[10,10],
        #                                                 criterion=criterion1)#,
        #                                                 # criterion2=criterion2)
    return model
        


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd  

 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Late_Fusion')
    args= parser.parse_args()
    return args

if __name__ == '__main__':
    print("Training Starting!")
    args = parse_args()
    # 获取当前时间
    now = datetime.now()
    # 格式化时间输出为"年-月-日 时:分"
    Exp_date = now.strftime("%Y_%m_%d %H_%M")
    # 数据: 1min专注 + 1min不专注 + 1min专注 + 1min不专注，前
    # D:\USTC\Program\metaBCI\github\0818\MetaBCI-USTChopin-Atten-CSP\MetaBCI-USTChopin-Atten\demos\brainflow_demos\Train_data0.npy两组训练，后两组测试
    # Atten_EEG = np.random.randn(4, 60*fs_eeg, 30)
    Atten_train = np.load("./Train_SHUK_0822_1_data.npy",allow_pickle=True)  #注意路径
    Atten_test = np.load("./Test_SHUK_0822_1_data.npy", allow_pickle=True)
    # 将采集数据提取成(4, 60*fs_eeg, 30)格式
    Atten_EEG = get_rawdata_atten(Atten_train, Atten_test)
    # print(Atten_EEG[0])
    win_len = 2 * fs_eeg
    step = int(0.5 * fs_eeg)
    # 切片，划分训练集测试机
    train_data, train_label, test_data, test_label = prepare_dataset_atten(win_len, step, Atten_EEG)
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    # 提取CSP特征，用LDA分类
    lda = LDA()
    csp = CSP()
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    CSP_train = csp.fit_transform(train_data, train_label)
    lda.fit(CSP_train, train_label)
    dump(csp, './' + Exp_date + '_{}_model.joblib'.format('CSP'))
    dump(lda, './' + Exp_date + '_{}_model.joblib'.format('LDA'))
    # csp = load('./' + '2024_08_18 17_16_CSP_model.joblib')
    # lda = load('./' + '2024_08_18 17_16_LDA_model.joblib')
    CSP_test = csp.transform(test_data)
    y_pred = lda.predict(CSP_test)
    acc = lda.score(CSP_test, test_label)
    print('Acc: ', acc)
    # test_data = np.random.randn(1, 30, win_len)
    # CSP_test = csp.transform(test_data)
    # y_pred = lda.predict(CSP_test)
    # print('y_pred', y_pred )

    # experiment_data = np.load("D:/USTC/Program/metaBCI/LuHK_2024_08_15 13_23_data.npy",allow_pickle=True)
    # experiment_data = np.load("ShuK_0819_data.npy",allow_pickle=True)
    # feature = []
    # EEG_All,EMG_All,Label_1_All,Label_2_All = get_rawdata(experiment_data)
    # length_eeg,length_emg,step = 300,200,100
    # Train_data,train_label,train_label_2,Test_data,test_label,test_label_2 = prepare_dataset(length_eeg,length_emg,step,EEG_All,EMG_All,Label_1_All,Label_2_All)
    # Train_data,train_label,train_label_2 = Feature_Engineering(Train_data,train_label,train_label_2)


    # print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))
    # print("train_label_2.shape:",np.array(train_label_2).shape,np.unique(np.array(train_label_2), return_counts=True))


    # # 加载模型
    # # lda_loaded = load('./'+Exp_date+'_lda_model.joblib')
    # Test_data,test_label,test_label_2 = Feature_Engineering(Test_data,test_label,test_label_2)
    # print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))
    # print("test_label_2.shape:",np.array(test_label_2).shape,np.unique(np.array(test_label_2), return_counts=True))

    # feature.append({
    #             'train_data': Train_data,
    #             'train_label': train_label,
    #             'train_label_2': train_label_2,
    #             'test_data': Test_data,
    #             'test_label': test_label,
    #             'test_label_2': test_label_2
    #             })

    # # np.save('D:/USTC/Program/metaBCI/MotorExecutionBCI/Features/'+
    # #         'ShuK' +'_'+'2024_07_21 15_04_data'+'_features_400_100.npy',feature)

    # # # 使用加载的模型进行预测（这里用测试集作为例子）#

    # # # np.hstack((np.array(Test_data[0]).reshape(np.array(Test_data[0]).shape[0],-1),np.array(Test_data[1]).reshape(np.array(Test_data[0]).shape[0],-1)))
    # # feature  = np.load('D:/USTC/Program/metaBCI/MotorExecutionBCI/Features/'+
    # #         'ShuK' +'_'+'2024_07_21 15_04_data'+'_features_400_100.npy',allow_pickle=True)
    # Train_data,train_label,train_label_2 = feature[0]['train_data'],feature[0]['train_label'],feature[0]['train_label_2']
    # Test_data,test_label,test_label_2 = feature[0]['test_data'],feature[0]['test_label'],feature[0]['test_label_2']

    # train_data = np.array(Train_data[1])
    # testdata =  np.array(Test_data[1]) #
    # # train_data = np.expand_dims(np.array(Train_data[0]),axis = 1)
    # # testdata =  np.expand_dims(np.array(Test_data[0]),axis = 1) #
    # # testdata = np.hstack((testdata,np.fft.fft(testdata,axis=2).real))
    # # testdata = np.fft.fft(testdata,axis=2).real  np.fft.fft(testdata,axis=2).imag,
    # # train_data = np.array(Train_data[0])#np.hstack((list(train_data[0]),list(train_data[1])))
    # # testdata_eeg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # # testdata = np.array(Test_data[1])
    # # train_data = np.array(Train_data[1])#np.hstack((list(train_data[0]),list(train_data[1])))
    # # testdata_emg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # # testdata = testdata_eeg#np.hstack((testdata_eeg,testdata_emg))#


    # model_type = args.model  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
    # print("model_type:",model_type)

    # if model_type == 'SKLDA':  ##调用metabci已有算法
    #     Trained_model = train_PR_model(train_data,train_label,train_label_2,Exp_date,type=model_type)
    #     y_pred = Trained_model.transform(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
    #     resutls = []
    #     for decision in y_pred:
    #         if decision > -2 and decision < -1:
    #             resutls.append(0)
    #         elif decision > -1 and decision < 0:
    #             resutls.append(1)
    #         elif decision > 0 and decision < 1:
    #             resutls.append(2)
    #         elif decision > 1 and decision <2:
    #             resutls.append(3)
    #         elif decision > 2 and decision <3:
    #             resutls.append(4)
    #         elif decision > 3 and decision <4:
    #             resutls.append(5)
    #     # print("y_pred",resutls,len(resutls))
    #     # print("test_label", test_label,len(test_label))
    #     # print('SKLDA分类正确率为：',np.sum(np.where(np.array(resutls)-np.array(test_label),0,1))/len(test_label)*100,'%')
    #     accuracy_loaded = accuracy_score(test_label, resutls)
    #     print(f"Accuracy with {model_type} model: {accuracy_loaded:.4f}")
    #     del Trained_model

    # if model_type == 'LDA' or model_type == 'SVM' or model_type == 'RandomForest':
    #     Trained_model,Trained_model_2 = train_PR_model(train_data,train_label,train_label_2,Exp_date,type=model_type)
    #     y_pred = Trained_model.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
    #     accuracy_loaded = accuracy_score(test_label, y_pred)
    #     cm = confusion_matrix(test_label, y_pred)

    #     y_pred_2 = Trained_model_2.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
    #     accuracy_loaded_2 = accuracy_score(test_label_2, y_pred_2)
    #     cm_2 = confusion_matrix(test_label_2, y_pred_2)
    #     dump(Trained_model, './'+Exp_date+'_{}_model.joblib'.format(model_type))
    #     dump(Trained_model_2, './'+Exp_date+'_{}_model_attention.joblib'.format(model_type))
    #     del Trained_model,Trained_model_2

    # elif model_type == 'Late_Fusion':
    #     train_data = Train_data
    #     testdata =  Test_data #
    #     print("Test_data:",type(Test_data[0]),type(Test_data[1]))

    #     Trained_model = train_PR_model(train_data,train_label,train_label_2,Exp_date,type=model_type)
    #     test_data = [torch.tensor(testdata[0],dtype=torch.float),
    #                                 torch.tensor(testdata[1],dtype=torch.float)]
    #     test_label = np.array(test_label)
    #     test_label_2 = np.array(test_label_2)
    #     y_pred = Trained_model.predict([test_data[0], test_data[1]])
    #     y_pred = np.array(y_pred)

    #     # test_loader1 = DataLoader(dataset=test_dataset,
    #     #                         batch_size=16,
    #     #                         shuffle=False)
    #     # y_pred,y_pred_attention, Test_Label,Test_Label_2 = test_Fusion(Trained_model,test_loader1)
    #     accuracy_loaded = accuracy_score(test_label, y_pred)
    #     cm = confusion_matrix(test_label, y_pred)
    #     print("confusion_matrix:",'\n',cm)
    #     print(f"Accuracy with {model_type} model: {accuracy_loaded:.4f}")
    #     # accuracy_loaded_2 = accuracy_score(test_label_2, y_pred_attention)
    #     # cm_2 = confusion_matrix(test_label_2, y_pred_attention)
    #     # dump(Trained_model, './'+Exp_date+'_{}_model.joblib'.format(model_type))
    #     del Trained_model


    # # 假设lda是已经训练好的LDA模型


 
