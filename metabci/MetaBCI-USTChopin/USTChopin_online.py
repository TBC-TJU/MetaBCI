import time
import random
import pygame
from pygame.locals import *
#from pylsl import StreamInfo, StreamOutlet
from neuracle_lib.dataServer import dataserver_thread
import numpy as np
from datetime import datetime
import serial
from neuracle_lib.triggerBox import TriggerBox,TriggerIn,PackageSensorPara
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import dump,load
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import emg_features 
from myModels import MLP_Test,Unimodal,Late_Fusion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import socket
from collections import Counter


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LDA')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default= 8712)
    
    args= parser.parse_args()
    return args
args = parse_args()
hostname = args.ip
port = args.port
M_type = args.model  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
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

# neuracle = dict(device_name = 'Neuracle',hostname = '127.0.0.1',port = 8712,
#                 srate = 1000,chanlocs = ['FPz','AF3'],n_chan = 2)

dsi = dict(device_name = 'DSI-24',hostname = '127.0.0.1',port = 8844,
            srate = 300,chanlocs = ['P3','C3','F3','Fz','F4','C4','P4','Cz','CM','A1','Fp1','Fp2','T3','T5','O1','O2','X3','X2','F7','F8','X1','A2','T6','T4','TRG'],n_chan = 25)

device = [neuracle,dsi]

# s_com = serial.Serial('COM6', 115200, timeout = None) #com6 替换为实验主机的端口号（详见硬件安装）

### pay attention to the device you used
target_device = device[0]
srate = target_device['srate']
#print('!!!! The type of device you used is %s'%target_device['device_name'])

## init dataserver
time_buffer = 5 # second
thread_data_server = dataserver_thread(threadName='data_server', device=target_device['device_name'], n_chan=target_device['n_chan'],
                                        hostname=target_device['hostname'], port= target_device['port'],srate=target_device['srate'],t_buffer=time_buffer)
thread_data_server.Daemon = True
notconnect = thread_data_server.connect()
if notconnect:
    print(0)# 
    #raise TypeError("Can't connect recorder, Please open the hostport ")
else:
    thread_data_server.start()
    print('Data server connected')

    
 
# 创建一个LSL流来发送实验标记
#info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
#outlet = StreamOutlet(info)

# 初始化 Pygame,获取显示器信息,并获取当前显示器的分辨率
pygame.init()
screen_info = pygame.display.Info()
current_width = screen_info.current_w
current_height = screen_info.current_h
  
# 设置屏幕大小和字体
# screen_width = 800
# screen_height = 600
# screen = pygame.display.set_mode((screen_width, screen_height))
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.display.set_caption('MetaBCI_BrainStim_USTChopin')
font = pygame.font.Font(None, 200)
font_size = 74
font_Chinese = pygame.font.Font('C:/Windows/Fonts/STKAITI.ttf', font_size)  # STKAITI华文正楷；none使用默认字体
small_font = pygame.font.Font(None, 36)

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# 图片地址
# path = 'D:/USTC/Program/metaBCI/MetaBCI-master/MetaBCI-master/metabci/brainstim/textures'
path = './Stim_images/'
image_cross_large = path+'/cross-large.png'  # 替换为你的图片路径
image_L0R0 = path+'/L0R0.jpg'    # 替换为你的图片路径
image_L1R1 = path+'/L1R1.jpg'    # 替换为你的图片路径
image_L2R2 = path+'/L2R2.jpg'    # 替换为你的图片路径
image_L3R3 = path+'/L3R3.jpg'    # 替换为你的图片路径
image_L4R4 = path+'/L4R4.jpg'    # 替换为你的图片路径
image_L5R5 = path+'/L5R5.jpg'    # 替换为你的图片路径
image_L6R6 = path+'/L6R6.jpg'    # 替换为你的图片路径


# 定义显示文本函数
def display_text(text, fonts, color, screen, x, y):
    text_surface = fonts.render(text, True, color)
    rect = text_surface.get_rect()
    rect.center = (x, y)
    screen.blit(text_surface, rect)
    pygame.display.flip()

# 定义显示图片函数
def display_image(image_path, color, screen):
    # 填充背景色
    screen.fill(color)

    # 获得图片及其尺寸
    image = pygame.image.load(image_path)
    image_width, image_height = image.get_size()

    # 将图片居中显示
    x = (current_width - image_width) // 2
    y = (current_height - image_height) // 2 

    # 将图片绘制到窗口上
    screen.blit(image, (x, y))
    pygame.display.flip()

# 定义显示图片函数
def display_text_image(text,fonts,image_path, color, screen):
    # 填充背景色
    screen.fill(color)

    # 获得图片及其尺寸
    image = pygame.image.load(path+image_path+'.jpg')
    image_width, image_height = image.get_size()

    # 将图片居中显示
    x = (current_width - image_width) // 2
    y = (current_height - image_height) // 2 
    # 将图片绘制到窗口上
    screen.blit(image, (x, y+100))
 
    #显示提示词
    text_surface = fonts.render(text, True, BLACK)
    rect = text_surface.get_rect()
    rect.center = (x+500, y-10) #左上角为原点，向右为x正方向，向下为y轴正方向
    screen.blit(text_surface, rect)   

    #更新显示
    pygame.display.flip()

# 设置实验参数
num_repeats = 1  # 总试次数 = num_repeats * trials
rest_time = 0.5    # 休息时间（秒）
prepare_time = 0.6  # 休息时间（秒）
imagine_time = 0.2  # 想象时间（秒）

# 初始化实验数据记录
experiment_data = []

triggerin = TriggerIn("COM7")
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

        eeg_filtered = np.transpose(eeg_filtered,(1,0)) - np.mean(eeg_filtered[:,:1000],1).reshape(1,-1) 
        # print("eeg_filtered2:",eeg_filtered.shape,voting_num)

        EEG_All.append(np.transpose(eeg_filtered,(1,0))[:,-int(imagine_time*1000*(voting_num+1))-1:-int(imagine_time*1000*voting_num)-1])
        # EEG_All.append(eeg_filtered[:,2000:])

        emg_filtered = signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)
        TEMP = np.mean(emg_filtered[:,:1000],1).reshape(1,-1)
        emg_filtered = np.transpose(emg_filtered,(1,0)) - TEMP
        EMG_All.append(np.abs(np.transpose(emg_filtered,(1,0))[:,-int(imagine_time*1000*(voting_num+1))-1:-int(imagine_time*1000*voting_num)-1]))
        # EMG_All.append(emg_filtered[:,2000:])

    # print("data.shape:",data.shape)
    # print("EEG_All.shape:",len(EEG_All))
    # print("EMG_All.shape:",len(EMG_All))
    return EEG_All,EMG_All


def segmentation(Rawdata,Label,length,step):
    window_length = length  #ms
    step = step #ms
    data = []
    label = []
    for i,rawdata in enumerate(Rawdata):
        point = 0
        while (point+window_length-1<rawdata.shape[1]):
            data.append(rawdata[:,point:point+window_length])
            point = point + step
            label.append(Label[i])
    #print("number of samples:",len(data))

    return data,label

def prepare_dataset(length,step,EEG_All,EMG_All,Label_All):
    train_eeg,train_label = segmentation(EEG_All[0:10*4],Label_All[0:10*4],length,step)#前4轮数据做训练数据
    train_emg,train_label = segmentation(EMG_All[0:10*4],Label_All[0:10*4],length,step)

    test_eeg,test_label = segmentation(EEG_All[10*4:],Label_All[10*4:],length,step)
    test_emg,test_label = segmentation(EMG_All[10*4:],Label_All[10*4:],length,step)
    

    train_data = [train_eeg,train_emg]
    test_data = [test_eeg,test_emg]

    return train_data,train_label,test_data,test_label


def extract_emg_feature(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(eval(str(func))(x[i,:]))
    res =np.vstack(res)
    return res

def Feature_Engineering(train_data,train_label):
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
    feature_list = ['ssc','wl','mean','rms','arc','log','mnf_MEDIAN_POWER']#]#,'mav','var']#,'psr']#,'arc','arc',
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
    # print("Feature Engineering is done!")
    return [eeg_feature,emg_feature],train_label

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

def testing(M_type,Test_data,test_label):
    # 加载模型
    model_type = M_type  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
    Test_data,test_label = Feature_Engineering(Test_data,test_label)

    # 使用加载的模型进行预测（这里用测试集作为例子）#
    testdata =  np.array(Test_data[1]) #
    # np.hstack((np.array(Test_data[0]).reshape(np.array(Test_data[0]).shape[0],-1),np.array(Test_data[1]).reshape(np.array(Test_data[0]).shape[0],-1)))
    
    # testdata = np.hstack((testdata,np.fft.fft(testdata,axis=2).real))
    # testdata = np.fft.fft(testdata,axis=2).real  np.fft.fft(testdata,axis=2).imag,
    # train_data = np.array(Train_data[0])#np.hstack((list(train_data[0]),list(train_data[1])))
    # testdata_eeg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # testdata = np.array(Test_data[1])  
    # train_data = np.array(Train_data[1])#np.hstack((list(train_data[0]),list(train_data[1])))
    # testdata_emg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # testdata = testdata_eeg#np.hstack((testdata_eeg,testdata_emg))#
    #print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))
    if model_type in {'LDA','SVM','RandomForest'}:
        # print("LDA ERRO:",'singular')
        y_pred = Trained_model.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
        y_pred = y_pred.tolist()
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
            # print("Test_data:",type(Test_data[0]),type(Test_data[1]))
            
            testdata1 =  np.array(Test_data[0]) # 
            testdata2 =  np.array(Test_data[1]) # 
            # testdata1 =  Test_data[0] # 
            # testdata2 =  Test_data[1] # 
            # print("testdata1 ERRO:",testdata1.shape,testdata2.shape)

            # test_dataset = TensorDataset(torch.tensor(testdata1,dtype=torch.float),
            #                              torch.tensor(testdata2,dtype=torch.float), 
            #                              torch.tensor(np.array(test_label),dtype=torch.long))
            # print("test_dataset ERRO:")
            
            # test_loader1 = DataLoader(dataset=test_dataset,
            #                         batch_size=1,
            #                         shuffle=False)
            # y_pred, Test_Label = test_Fusion(Trained_model,test_loader1)  
            testdata1 = torch.tensor(testdata1,dtype=torch.float).to(device)
            testdata2 = torch.tensor(testdata2,dtype=torch.float).to(device)

            outputs,x5,a5 = Trained_model(testdata1,testdata2)  
            # print("outputs:",outputs.shape,outputs)

            _, y_pred = torch.max(outputs.data, 1)
            # print("predicted:",y_pred)
            y_pred = y_pred.tolist()
            # accuracy_loaded = accuracy_score(test_label, y_pred)

            # cm = confusion_matrix(test_label, y_pred)
            # 假设Unimodal是已经训练好的Unimodal模型
            # dump(Trained_model, './'+Exp_date+'_LateFusion_model.joblib')
            # print(f"Accuracy with Unimodal model: {accuracy_loaded:.4f}")    
            # print("confusion_matrix:",'\n',cm)
        # y_pred, Test_Label = np.array(y_pred), np.array(Test_Label)
        # accuracy_loaded = accuracy_score(Test_Label, y_pred)

        # accuracy_loaded =  (y_pred == Test_Label).sum().item()/len(Test_Label)
        # y_pred = y_pred.item()

    # cm = confusion_matrix(test_label, y_pred)
    # print(f"Accuracy with loaded model: {accuracy_loaded:.4f}")    
    # print("confusion_matrix:",'\n',cm)
    print("Truth, prediction:",test_label,y_pred )
    return y_pred
# 主实验循环





if __name__ == '__main__':
    #("Training Starting!")
 
    soc = socket.socket()
    soc.connect(('127.0.0.1',65432))    
    print("model_type:",M_type)
    if M_type == "LDA":
        Trained_model = load('./'+'2024_08_03 17_04_LDA_model.joblib')
    elif M_type == "Late_Fusion":
        Trained_model = load('./'+'2024_08_03 17_01_Late_Fusion_model.joblib')
    Num_voting = 3
    running = True
    Prediction_all = []
    Label = []
    for repeat in range(num_repeats):
        # 随机选择一个想象动作
        # tasks =  list(random.sample(range(1, 7), 6))
        tasks =  list([5,3,5,3,5,3,1,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1,2,2,4,4,3,1,5,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1])

        num_task = len(tasks)
        print("num_task:",num_task)
        i = 0
        for task in tasks:
            i = i+ 1
            # 检查退出事件
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    break
                # 检测键盘按键事件
                elif event.type == pygame.KEYDOWN:  #有键被按下
                    if event.key == pygame.K_SPACE: #空格键被按下 
                        running = False   
                        break 
            if not running:
                break
    
            # # 显示固定点，提示被试准备
            # screen.fill(BLACK)
            # display_text('KEY'+" " +str(task), font, WHITE, screen, current_width//2, current_height//2-150)
            # display_text('+', font, WHITE, screen, current_width//2, current_height//2)
            # time.sleep(prepare_time)
            
            # 显示想象指令，并发送相应的标记
            screen.fill(BLACK)
            image = 'L'+str(task)+'R'+str(task)
            display_text_image('KEY'+" " +str(task),font_Chinese,image, WHITE, screen)
            marker = task    
            # s.write([task])
            # outlet.push_sample([task])
            # start_time = time.time()  # 记录开始时间
            time.sleep(imagine_time+1)
    
            flagstop = False
            try:
                while not flagstop: # get data in one second step
                    # s_com.write([task])
                    pred = []
                    data = thread_data_server.get_bufferData()
                    # EEG_All,EMG_All  = get_rawdata(data)
                    # EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)
                    # print("EEG_All,EMG_All:",EEG_All.shape,EMG_All.shape)
                    for voting in range (Num_voting):
                        EEG_All,EMG_All  = get_rawdata(data,voting)
                        # print("EEG_All,EMG_All:",EEG_All.shape,EMG_All.shape)

                        EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)
                        # print("EEG_All2,EMG_All:",EEG_All.shape,EMG_All.shape)
                        
                        pred.append(testing(M_type,[EEG_All,EMG_All],task)[0])
                        # print("pred[i]:",pred[voting])
                    prediction = Counter(pred).most_common(1)[0][0]
                    soc.send(str(prediction).encode("UTF-8"))

                    # msg = input("请输入发送给服务端的消息：")
                    # if "exit" == msg:
                    #     break
                    # soc.send(msg.encode("UTF-8"))
                    # data = soc.recv(1024).decode("UTF-8")
                    # print(f"服务端发来的消息：{data}")


                    # print("prediction:",prediction)
                    Prediction_all.append(prediction)
                    Label.append([np.array(task)])
                    # Label = [Label,task]
                    time.sleep(imagine_time+1)

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
        
            # 确保被试有足够的休息时间
        #     screen.fill(BLACK)
        #     display_text('休息', font_Chinese, WHITE, screen, current_width//2, current_height//2)
        #     time.sleep(rest_time) 

        #     flagstop = False
        #     try:
        #         while not flagstop: # get data in one second step   
        #             # s_com.write([6])
        #             data = thread_data_server.get_bufferData()
        #             EEG_All,EMG_All  = get_rawdata(data,0)
        #             EEG_All,EMG_All = np.array(EEG_All),np.array(EMG_All)

        #             # prediction = testing(M_type,[EEG_All,EMG_All],0)

        #             # Prediction_all.extend(prediction.tolist())
        #             # Label.extend([np.array(task)])
        #             # Label = [Label,0]

        #             flagstop = True
        #     except:
        #         pass
        #     # 保存实验数据
        #     experiment_data.append({
        #         'data': data,
        #         'trial': repeat*num_task + i,
        #         'marker': 0
        #     })
        # # 确保被试在每一轮动作后有足够的休息时间
        # screen.fill(BLACK)

        # display_text('间期休息', font_Chinese, WHITE, screen, current_width//2, current_height//2)
        # time.sleep(rest_time+15) 
    soc.send(str(7).encode('UTF-8'))
    Truth = np.array(Label).squeeze()
    print("Truth,prediction:",Truth,Prediction_all)
    accuracy_loaded = accuracy_score(Truth.tolist(), Prediction_all)
    cm = confusion_matrix(Label, Prediction_all)
    #print(f"Accuracy with LDA model: {accuracy_loaded:.4f}")    
    print(f"Accuracy with loaded model: {accuracy_loaded:.4f}")    
    print("confusion_matrix:",'\n',cm)

    thread_data_server.stop()
    # 结束实验
    pygame.quit()

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

