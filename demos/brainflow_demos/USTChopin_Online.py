import time
import random
import pygame
from pygame.locals import *
#from pylsl import StreamInfo, StreamOutlet  
import numpy as np
from datetime import datetime
import serial
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
import socket
from collections import Counter
from metabci.brainflow.amplifiers import Neuracle, Marker
from metabci.brainflow.workers import ProcessWorker
import os
import argparse
from psychopy import core

# from metabci.brainda.algorithms.feature_engineering import emg_features 
from metabci.brainflow.amplifiers import Neuracle
from metabci.brainda.algorithms.deep_learning.Fusion import Late_Fusion_Attention,Unimodal,Late_Fusion
from skorch.callbacks import Checkpoint
from metabci.brainda.algorithms.feature_engineering.ExtractingFeatures import Extracting_Feature
from metabci.brainstim.paradigm import PlayPiano

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Late_Fusion')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default= 8712)
    
    args= parser.parse_args()
    return args



# 设置实验参数
num_repeats = 1  # 总试次数 = num_repeats * trials
rest_time = 0.5    # 休息时间（秒）
prepare_time = 0.6  # 休息时间（秒）
imagine_time = 0.2  # 想象时间（秒）

# 初始化实验数据记录
experiment_data = []

# triggerin = TriggerIn("COM7")
# triggerbox = TriggerBox("COM1")
# isTriggerIn = True
# isTriggerBox = False

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




def get_rawdata(experiment_data,voting_num):
    EEG_All = []
    EMG_All = []
    Label_All = []
    # 打印实验数据
    # for data in experiment_data:

    raw = experiment_data
    # print("raw:",raw.shape)
    raw = signal.filtfilt(b_notch, a_notch,raw,axis=1)
    list_eeg = range(0,30)#(7,20)#(2,25)#(7,8,9,10,12,13,14,)#
    list_emg = (35,40,39,33,36,51,56,53,46,48)#,)#
    if raw[list_eeg,:].shape[1]>600:
        # eeg_filtered = signal.filtfilt(b11, a11,raw[list_eeg,:],axis=1)
        eeg_filtered = raw[list_eeg,:]
        # print("eeg_filtered:",eeg_filtered.shape)

        eeg_filtered = np.transpose(eeg_filtered,(1,0)) #- np.mean(eeg_filtered[:,:1000],1).reshape(1,-1) 
        # print("eeg_filtered2:",eeg_filtered.shape,voting_num)

        EEG_All.append(np.transpose(eeg_filtered,(1,0))[:,-int(imagine_time*1000*(voting_num+1))-1:-int(imagine_time*1000*voting_num)-1])
        # EEG_All.append(eeg_filtered[:,2000:])

        emg_filtered = signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)
        TEMP = np.mean(emg_filtered[:,:1000],1).reshape(1,-1)
        emg_filtered = np.transpose(emg_filtered,(1,0)) #- TEMP
        EMG_All.append(np.abs(np.transpose(emg_filtered,(1,0))[:,-int(imagine_time*1000*(voting_num+1))-1:-int(imagine_time*1000*voting_num)-1]))
        # EMG_All.append(emg_filtered[:,2000:])

    print("data.shape:",data.shape)
    print("EEG_All.shape:",len(EEG_All))
    print("EMG_All.shape:",len(EMG_All))
    return EEG_All,EMG_All

def get_rawdata_step(experiment_data,window_eeg,window_emg,step,voting_num):
    EEG_All = []
    EMG_All = []
    Label_All = []
    # 打印实验数据
    # print("raw:" )
    raw = experiment_data#[-1]['data']
    # print("raw:",raw.shape)
    raw = signal.filtfilt(b_notch, a_notch,raw,axis=1)
    list_eeg = range(0,30)#(7,20)#(2,25)#(7,8,9,10,12,13,14,)#
    list_emg = (35,40,39,33,36,51,56,53,46,48)#,)#
    if raw[list_eeg,:].shape[1]>600:
        # eeg_filtered = signal.filtfilt(b11, a11,raw[list_eeg,:],axis=1)
        eeg_filtered = raw[list_eeg,:]
        # print("eeg_filtered:",eeg_filtered.shape)
        eeg_filtered = np.transpose(eeg_filtered,(1,0)) - np.mean(eeg_filtered[:,:1000],1).reshape(1,-1) 
        # print("eeg_filtered2:",eeg_filtered.shape,voting_num)

        emg_filtered = signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)
        TEMP = np.mean(emg_filtered[:,:1000],1).reshape(1,-1)
        emg_filtered = np.transpose(emg_filtered,(1,0)) - TEMP

        for voting_i in range(voting_num):
            EEG_All.append(np.transpose(eeg_filtered,(1,0))[:,-int(window_eeg+step*voting_i)-1:-int(step*voting_i)-1])
            # EEG_All.append(eeg_filtered[:,2000:])

            EMG_All.append(np.abs(np.transpose(emg_filtered,(1,0))[:,-int(window_emg+step*voting_i)-1:-int(step*voting_i)-1]))
            # EMG_All.append(emg_filtered[:,2000:])

    # print("data.shape:",EEG_All,EMG_All)
    # print("EEG_All.shape:",len(EEG_All))
    # print("EMG_All.shape:",len(EMG_All))
    return EEG_All,EMG_All,eeg_filtered[-2000:,:]

# def segmentation(Rawdata,Label,length,step):
#     window_length = length  #ms
#     step = step #ms
#     data = []
#     label = []
#     for i,rawdata in enumerate(Rawdata):
#         point = 0
#         while (point+window_length-1<rawdata.shape[1]):
#             data.append(rawdata[:,point:point+window_length])
#             point = point + step
#             label.append(Label[i])
#     #print("number of samples:",len(data))

#     return data,label

# def prepare_dataset(length,step,EEG_All,EMG_All,Label_All):
#     train_eeg,train_label = segmentation(EEG_All[0:10*4],Label_All[0:10*4],length,step)#前4轮数据做训练数据
#     train_emg,train_label = segmentation(EMG_All[0:10*4],Label_All[0:10*4],length,step)

#     test_eeg,test_label = segmentation(EEG_All[10*4:],Label_All[10*4:],length,step)
#     test_emg,test_label = segmentation(EMG_All[10*4:],Label_All[10*4:],length,step)
    

#     train_data = [train_eeg,train_emg]
#     test_data = [test_eeg,test_emg]

#     return train_data,train_label,test_data,test_label

##调用新增的特征提取功能
FeatureEngineering = Extracting_Feature(threshold  = 1e-5, EPS = 1e-5, fs=100, type='MEDIAN_POWER')
def extract_emg_feature(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(FeatureEngineering.predict(x[i,:],feature_name))   #调用Extracting_Feature
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

def test_Deeplearning(model,test_loader):
    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        Outcome = []
        Label = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Outcome.extend(predicted.tolist())
            Label.extend(labels.tolist())
    #print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return Outcome, Label

def test_Fusion(model,test_loader):
    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        Outcome = []
        Label = []
        for images,images2, labels in test_loader:
            print("qqimages,images2:",images.shape,images2.shape)

            images = images.to(device)
            images2 = images2.to(device)
            print("images,images2:",images.shape,images2.shape)

            labels = labels.to(device)
            outputs,x5,a5 = model(images,images2)
            print("outputs:",outputs.shape,outputs)

            _, predicted = torch.max(outputs.data, 1)
            print("predicted:",predicted)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Outcome.extend(predicted.tolist())
            Label.extend(labels.tolist())
    #print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return Outcome, Label

def testing(M_type,Test_data,test_label,test_label_2):
    # 加载模型
    model_type = M_type  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
    Test_data,test_label,test_label_2 = Feature_Engineering(Test_data,test_label,test_label_2)

    # 使用加载的模型进行预测（这里用测试集作为例子）#
    testdata =  np.array(Test_data[1]) #
    if model_type in {'LDA','SVM','RandomForest'}:
        # print("LDA ERRO:",'singular')
        y_pred = Trained_model.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
        y_pred = y_pred.tolist()
        y_pred_2 = Trained_model_2.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
        y_pred_2 = y_pred_2.tolist()

        # accuracy_loaded = accuracy_score(test_label.tolist(), y_pred.tolist())
        # accuracy_loaded = accuracy_score(test_label, y_pred)


    elif model_type in  {'MLP','Unimodal'}:
        test_dataset = TensorDataset(torch.tensor(testdata,dtype=torch.float),
                                       torch.tensor(np.array(test_label),dtype=torch.long))
        test_loader1 = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)
        y_pred, Test_Label = test_Deeplearning(Trained_model,test_loader1)  



    elif model_type =='Late_Fusion':
            testdata = Test_data
            test_data = [torch.tensor(testdata[0],dtype=torch.float),
                                        torch.tensor(testdata[1],dtype=torch.float)]
            test_label = np.array(test_label)
            test_label_2 = np.array(test_label_2)
            y_pred = Trained_model.predict([test_data[0], test_data[1]])
            y_pred = np.array(y_pred)
            
 
    print("Gesture Truth, prediction:",test_label,y_pred)
    # print("Attention Truth, prediction:",test_label_2,y_pred_2 )
    return y_pred
# 主实验循环





if __name__ == '__main__':
    #("Training Starting!")
    args = parse_args()
    hostname = args.ip
    port = args.port
    M_type = args.model  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
    srate = 1000
 
    # 58+1 channels;  与Neuracle recorder中的dataservice Montage顺序保持一致
    # 其中30 EEG，28 sEMG, 1 TRG
    neuracle = dict(device_name = 'Neuracle',hostname = hostname, port = int(port),
                    srate = 1000,chanlocs = ['FP1','FP2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
                                                'CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO3','PO4','O1','Oz','O2',
                                                'ECG','PO7','TP7','P5','FT7','FC3','F5','AF7','AF3','F1','C5','CP3',
                                                'POz','PO6','PO5','PO8','P6','TP8','C6','CP4','FT8','FC4','F6','AF8','F2','FCz','AF4','FPz','TRG'],
                                                n_chan = 59)
            # neuracle = dict(device_name = 'Neuracle',hostname = '127.0.0.1',port = 8712,
            #                 srate = 1000,chanlocs = ['FP1'(0),'FP2'(1),'F7'(2),'F3'(3),'Fz'(4),'F4'(5),'F8'(6),'FC5'(7),'FC1'(8),'FC2'(9),'FC6'(10),'T7'(11),'C3'(12),'Cz'(13),'C4'(14),'T8'(15),
            #                                             'CP5'(16),'CP1'(17),'CP2'(18),'CP6'(19),'P7'(20),'P3'(21),'Pz'(22),'P4'(23),'P8'(24),'PO3'(25),'PO4'(26),'O1'(27),'Oz'(28),'O2'(29),
            #                                             'ECG'(30),'PO7'(31),'TP7'(32),'P5'(33),'FT7'(34),'FC3'(35),'F5'(36),'AF7'(37),'AF3'(38),'F1'(39),'C5'(40),'CP3'(41),
            #                                             'POz'(42),'PO6'(43),'PO5'(44),'PO8'(45),'P6'(46),'TP8'(47),'C6'(48),'CP4'(49),'FT8'(50),'FC4'(51),'F6'(52),'AF8'(53),'F2'(54),'FCz'(55),'AF4'(56),'FPz'(57),'TRG'(58)],
            #                                             n_chan = 59)
 
    dsi = dict(device_name = 'DSI-24',hostname = '127.0.0.1',port = 8844,
                srate = 300,chanlocs = ['P3','C3','F3','Fz','F4','C4','P4','Cz','CM','A1','Fp1','Fp2','T3','T5','O1','O2','X3','X2','F7','F8','X1','A2','T6','T4','TRG'],n_chan = 25)

    device = [neuracle,dsi]

    target_device = device[0]
    srate = target_device['srate']
    print('!!!! The type of device you used is %s'%target_device['device_name'])

    ## init dataserver
    time_buffer = 5 # second
    thread_data_server = Neuracle(threadName='data_server', device=target_device['device_name'], n_chan=target_device['n_chan'],
                                            hostname=target_device['hostname'], port= target_device['port'],srate=target_device['srate'],t_buffer=time_buffer)
    thread_data_server.Daemon = True
    notconnect = thread_data_server.connect()
    if notconnect:
        print(0)# 
        #raise TypeError("Can't connect recorder, Please open the hostport ")
    else:
        thread_data_server.start()
        print('Data server connected')


## init dataserver

    # time_buffer = 5 # second
    # thread_data_server = Neuracle_Update(device_address = (hostname,port), srate=1000, num_chans=59, t_buffer=time_buffer)
    # #thread_data_server.Daemon = True
    # thread_data_server.connect_tcp()
    # # if notconnect:
    # #     print(0)# 
    # #raise TypeError("Can't connect recorder, Please open the hostport ")
    # print('Data server connected')
    # thread_data_server.recv()
    # print('!!!! The type of device you used is Neuracle')
    

    
# 创建一个LSL流来发送实验标记
#info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
#outlet = StreamOutlet(info)


  
# 设置屏幕大小和字体
# screen_width = 800
# screen_height = 600
# screen = pygame.display.set_mode((screen_width, screen_height))

# 定义颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    soc = socket.socket()
    soc.connect(('127.0.0.1',65432))    

    Num_voting = 3
    window_eeg,window_emg,step = 200,200,100
    running = True
    Prediction_all = []
    Attention_all = []
    Label = []
    # experiment = USTChopin(mode='online')
    experiment = PlayPiano(ip='127.0.0.1', port=8712, device_idx=0, mode ='online_Testing')

    # 载入专注度模型
    csp = load('./' + '2024_08_22 11_39_CSP_model.joblib')
    Atten_model = load('./' + '2024_08_22 11_39_LDA_model.joblib')

    print("model_type:",M_type)
    if M_type == "LDA":
        Trained_model = load('./'+'2024_08_21 15_33_LDA_Gmodel.joblib')
        Trained_model_2 = load('./'+'2024_08_19 13_10_LDA_model_attention.joblib')


    elif M_type == "Late_Fusion":
        # Trained_model = load('./'+'2024_08_13 19_09_Late_Fusion_model.joblib')
        Trained_model = Late_Fusion(dropout=0.15,input_dim1=window_eeg*30,
                                    channel_eeg=7,input_dim2=6*10,
                                    channel_emg=3,classes=6)
        # mymodel.load_state_dict(checkpoint)
         #mymodeleval()
        Trained_model.initialize()
        cp = Checkpoint(dirname="checkpoints/2244584306144")
        Trained_model.load_params(checkpoint=cp)
 
    for repeat in range(num_repeats):
        # 随机选择一个想象动作
        # tasks =  list(random.sample(range(1, 7), 6))
        tasks =  list([5,3,5,3,5,3,1,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1,2,2,4,4,3,1,5,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1])
        num_task = len(tasks)
        tasks_2 = [1]* num_task

        print("num_task:",num_task)
        i = 0
        for task in tasks:
            i = i+ 1
            # 检查退出事件
            experiment.display_image(image_path=f'./demos/brainstim_demos/Stim_images/picture{i}.jpg')
            core.wait(3*(imagine_time+0.8)/3)
            experiment.display_image(image_path=f'./demos/brainstim_demos/Stim_images/picture0.jpg')  
            core.wait((imagine_time+0.8)/3)
    
            flagstop = False
            try:
                while not flagstop: # get data in one second step
                    # s_com.write([task])
                    pred = []
                    # _ = experiment.collect_data(task, repeat, 1, 1, True)
                    # data = experiment.experiment_data #
                    # print("data:",data[-1]["data"])
                    data = thread_data_server.get_bufferData()
                    # EEG_All,EMG_All  = get_rawdata(data)
                    # EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)
                    # print("EEG_All,EMG_All:",EEG_All.shape,EMG_All.shape)

                    # for voting in range (Num_voting):
                    #     # EEG_All,EMG_All  = get_rawdata(data,voting)
                    #     # print("EEG_All,EMG_All:",EEG_All.shape,EMG_All.shape)

                    #     EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)
                    #     # print("EEG_All2,EMG_All:",EEG_All.shape,EMG_All.shape)
                        
                    #     pred.append(testing(M_type,[EEG_All,EMG_All],task)[0])
                        # print("pred[i]:",pred[voting])
                    
                    EEG_All,EMG_All,EEG_Atten  = get_rawdata_step(data,window_eeg,window_emg,step,Num_voting)
                    EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)
                    # print("EEG_Atten:",EEG_Atten.shape)

                    EEG_Atten = np.reshape(EEG_Atten,(1,30,2000))
                    # print("EEG_Atten:",EEG_Atten.shape)
                    # 对EEG数据预测专注度
                    CSP_test = csp.transform(EEG_Atten)
                    pred_2 = Atten_model.predict(CSP_test)

                    pred = testing(M_type,[EEG_All,EMG_All],task,1)
                    pred_attention = pred_2#Counter(pred_2).most_common(1)[0][0]
                    # print("pred_attention,pred_2:",pred_attention,pred_2)
                    if pred_attention == 1:
                        if task in pred:
                            prediction = task
                        else:
                            prediction = Counter(pred).most_common(1)[0][0]
                    else:
                        prediction = Counter(pred).most_common(1)[0][0]
                    #print("prediction:",prediction)
                    soc.send(str(prediction).encode("UTF-8"))
                    # msg = input("请输入发送给服务端的消息：")
                    # if "exit" == msg:
                    #     break
                    # soc.send(msg.encode("UTF-8"))
                    # data = soc.recv(1024).decode("UTF-8")
                    # print(f"服务端发来的消息：{data}")


                    # print("prediction:",prediction)
                    Prediction_all.append(prediction)
                    Attention_all.append(pred_attention)

                    Label.append([np.array(task)])
                    # Label = [Label,task]
                    core.wait(imagine_time+0.8)

                    flagstop = True
            except Exception as e:
                print(e)

            # 记录结束标记
            # outlet.push_sample([0])
            # end_time = time.time()  # 记录结束时间
            
            # 保存实验数据
            experiment_data.append({
                'data': data,
                'trial': repeat*num_task + i,
                'marker': task
            })
        
            
    soc.send(str(7).encode('UTF-8'))
    Truth = np.array(Label).squeeze()
    print("Gesture Truth,prediction 2:",Truth,Prediction_all)
    print("Attention Truth,prediction 2:",tasks_2,Attention_all)
    
    accuracy_loaded = accuracy_score(Truth.tolist(), Prediction_all)
    cm = confusion_matrix(Label, Prediction_all)
    print(f"Accuracy with loaded model: {accuracy_loaded:.4f}")    
    print("confusion_matrix:",'\n',cm)

    accuracy_loaded_2 = accuracy_score(tasks_2, Attention_all)
    cm_2 = confusion_matrix(tasks_2, Attention_all)
    print(f"Accuracy with loaded model: {accuracy_loaded_2:.4f}")    
    print("confusion_matrix:",'\n',cm_2)

    image = 'picture53'
    experiment.display_text_image(text=f'粉刷匠', image_path=f'./demos/brainstim_demos/Stim_images/picture53.jpg', color=(0, 0, 0), position=[(0, 0)])
    core.wait(30)
    # thread_data_server.stop()
    # 结束实验

    experiment.win.close()


    soc.close()


    # 获取当前时间
    now = datetime.now()

    # 格式化时间输出为"年-月-日 时:分"
    Exp_date = now.strftime("%Y_%m_%d %H_%M")
    Subject = 'ShuK'
    print("当前时间：", Exp_date)
    # np.save('D:/USTC/Program/metaBCI/'+Subject +'_'+Exp_date+'_online_data.npy',experiment_data)

    # 打印实验数据
    # for data in experiment_data:
        # print(data)

