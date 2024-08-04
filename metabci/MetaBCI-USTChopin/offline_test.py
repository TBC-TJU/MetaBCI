# import pyEDFlib
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
import emg_features 
from myModels import MLP_Test,Unimodal,Late_Fusion,EEG_model,EEG_model2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import random
import torch.nn.functional as F
from thop import profile

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
    Label_All = []
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
        elif raw[list_eeg,:].shape[1]==800:
            EEG_All.append(raw[list_eeg,:])
            # EEG_All.append(signal.filtfilt(b11, a11, raw[list_eeg,:],axis=1))
            EMG_All.append(np.abs(signal.filtfilt(b12, a12,raw[list_emg,:],axis=1)))
            # EMG_All.append(raw[list_emg,:])
 
        Label_All.append(np.array(data['marker']))
    print("data['data'].shape:",data['data'].shape)
    print("EEG_All.shape:",len(EEG_All))
    print("EMG_All.shape:",len(EMG_All))
    print("Label_All.shape:",np.array(Label_All).shape,np.unique(np.array(Label_All), return_counts=True))
    return EEG_All,EMG_All,Label_All

def segmentation(Rawdata,Label,length,step):
    window_length = length  #ms
    step = step #ms
    data = []
    label = []
    print("Label.shape:",np.array(Label).shape,np.unique(np.array(Label), return_counts=True))

    for i,rawdata in enumerate(Rawdata):
        point = 0
        while (point+window_length-1<rawdata.shape[1]):
            data.append(rawdata[:,point:point+window_length])
            point = point + step
            label.append(Label[i])
    print("number of samples:",len(data))
    # print("label.shape:",np.array(label).shape,np.unique(np.array(label), return_counts=True))
    return data,label

def prepare_dataset(length,step,EEG_All,EMG_All,Label_All):
    train_eeg,train_label = segmentation(EEG_All[0:10*4],Label_All[0:10*4],length,step)#前4轮数据做训练数据
    train_emg,train_label = segmentation(EMG_All[0:10*4],Label_All[0:10*4],length,step)

    test_eeg,test_label = segmentation(EEG_All[10*4:],Label_All[10*4:],length,step)
    test_emg,test_label = segmentation(EMG_All[10*4:],Label_All[10*4:],length,step)
    
    train_data = [train_eeg,train_emg]
    test_data = [test_eeg,test_emg]
    # print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))
    # print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))

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
    feature_list = ['ssc','wl','mean','rms','arc','log','mnf_MEDIAN_POWER']#]#,'mav','var']#,'psr']#,'arc'
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
    return [eeg_feature,emg_feature],train_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def trian_DeepLearning(model,learning_rate,train_loader,num_epochs,criterion):
    # 定义优化器
    for i,lr in enumerate(learning_rate):
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        total_step = len(train_loader)
        for epoch in range(num_epochs[i]):
            correct = 0
            total = 0
            Outcome = []
            Label = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                Outcome.append(predicted)
                Label.append(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Accuracy: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),100 * correct / total))
        print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    return model,Outcome, Label

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
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return Outcome, Label


def trian_Fusion(model,learning_rate,train_loader,num_epochs,criterion):#,criterion2):
    # 定义优化器
    for i,lr in enumerate(learning_rate):
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        total_step = len(train_loader)
        for epoch in range(num_epochs[i]):
            correct = 0
            total = 0
            Outcome = []
            Label = []
            for i, (images1, images2, labels) in enumerate(train_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs,x5,a5 = model(images1,images2)
                # x5,a5 = F.normalize(x5, dim=1),F.normalize(a5, dim=1)
                loss = criterion(outputs, labels)\
                    # + kd_loss(x5, a5, 1)\
                    # + criterion2(torch.cat([x5.unsqueeze(1),a5.unsqueeze(1)],1),labels)\

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                Outcome.append(predicted)
                Label.append(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Accuracy: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),100 * correct / total))
        print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    return model,Outcome, Label


def test_Fusion(model,test_loader):
    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        Outcome = []
        Label = []
        for images,images2, labels in test_loader:
            images = images.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            outputs,x5,a5 = model(images,images2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Outcome.extend(predicted.tolist())
            Label.extend(labels.tolist())
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return Outcome, Label

def train_PR_model(train_data,train_label,Exp_date,type):
    data = train_data#np.array(train_data[0])#np.hstack((list(train_data[0]),list(train_data[1])))
    
    label = train_label
    if type=="LDA":
        model = LDA()
        model.fit(np.reshape(data,(data.shape[0],-1)), label)

    elif type=="SVM":
        model = SVC(kernel='rbf')  # 可以选择不同的核函数，如'rbf', 'poly','linear'等
        model.fit(np.reshape(data,(data.shape[0],-1)), label)
 
    elif type=="RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(np.reshape(data,(data.shape[0],-1)), label)

    elif type=="MLP":
        setup_seed(9999)
        # 实例化模型
        model = MLP_Test(dropout=0.05,input_size=data.shape[2]*data.shape[3],
                         hiden_size=300,channel=data.shape[1],classes=6).to(device)
        img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
        img = img.to("cuda")
        flops, params = profile(model, inputs=(img,))
        print("flops:",flops)
        print("params:",params)
        print("MLP params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位， 
    
        criterion1 = nn.CrossEntropyLoss()
        # 数据加载器
        train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
                                       torch.tensor(np.array(label),dtype=torch.long))
        train_loader1 =  DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
                                                        train_loader=train_loader1,
                                                        num_epochs=[20,20],
                                                        criterion=criterion1)
    elif type == "Unimodal":
        setup_seed(9999)
        # 实例化模型
        model = Unimodal(dropout=0.05,input_dim=data.shape[2]*data.shape[3],
                         channel=data.shape[1],classes=6).to(device)
        img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
        img = img.to("cuda")
        flops, params = profile(model, inputs=(img,))
        print("flops:",flops)
        print("params:",params)
        print("Unimodal params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位， 
    
        criterion1 = nn.CrossEntropyLoss()

        # 数据加载器
        train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
                                       torch.tensor(np.array(label),dtype=torch.long))
        train_loader1 = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
                                                        train_loader=train_loader1,
                                                        num_epochs=[20,20],
                                                        criterion=criterion1)

    elif type == "EEG_model":
        setup_seed(9999)
        # 实例化模型
        model = EEG_model(dropout=0.05,input_size=data.shape[2]*data.shape[3],
                         hiden_size=300,channel=data.shape[1],classes=6).to(device)
        img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
        img = img.to("cuda")
        flops, params = profile(model, inputs=(img,))
        print("flops:",flops)
        print("params:",params)
        print("Unimodal params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位， 
    
        criterion1 = nn.CrossEntropyLoss()

        # 数据加载器
        train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
                                       torch.tensor(np.array(label),dtype=torch.long))
        train_loader1 = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
                                                        train_loader=train_loader1,
                                                        num_epochs=[20,20],
                                                        criterion=criterion1)        

    elif type == "EEG_model2":
        setup_seed(9999)
        # 实例化模型
        model = EEG_model2(dropout=0.05,input_dim1=data.shape[2]*data.shape[3],
                         channel_eeg=data.shape[1],classes=6).to(device)
        img = torch.randn(2,data.shape[1],data.shape[2],data.shape[3])
        img = img.to("cuda")
        flops, params = profile(model, inputs=(img,))
        print("flops:",flops)
        print("params:",params)
        print("Unimodal params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位， 
    
        criterion1 = nn.CrossEntropyLoss()

        # 数据加载器
        train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float32),
                                       torch.tensor(np.array(label),dtype=torch.long))
        train_loader1 = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        model,Train_Outcome, Train_Label = trian_DeepLearning(model,learning_rate=[0.01,0.001],
                                                        train_loader=train_loader1,
                                                        num_epochs=[20,20],
                                                        criterion=criterion1)          
    elif type == "Late_Fusion":
        setup_seed(9999)
        # 实例化模型dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes
        model = Late_Fusion(dropout=0.15,input_dim1=data[0].shape[2]*data[0].shape[3],
                         channel_eeg=data[0].shape[1],input_dim2=data[1].shape[2]*data[1].shape[3],
                         channel_emg=data[1].shape[1],classes=6).to(device)
        img = torch.randn(2,data[0].shape[1],data[0].shape[2],data[0].shape[3])
        img1 = torch.randn(2,data[1].shape[1],data[1].shape[2],data[1].shape[3])
        img1 = img1.to("cuda")
        img = img.to("cuda")
        flops, params = profile(model, inputs=(img,img1,))
        print("flops:",flops)
        print("params:",params)
        print("Late_Fusion params %.2f M| flops %.2f M" % (  params / (1000 ** 2), flops / (1000 ** 2)))#这里除以1000的平方，是为是为了化成M的单位， 
    
        criterion1 = nn.CrossEntropyLoss()
        criterion2 =  SupConLoss()
        # 数据加载器
        train_dataset = TensorDataset(torch.tensor(data[0],dtype=torch.float32),
                                      torch.tensor(data[1],dtype=torch.float32),
                                       torch.tensor(np.array(label),dtype=torch.long))
        train_loader1 = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        model,Train_Outcome, Train_Label = trian_Fusion(model,learning_rate=[0.01,0.001],
                                                        train_loader=train_loader1,
                                                        num_epochs=[20,20],
                                                        criterion=criterion1)#,
                                                        # criterion2=criterion2)        
    return model


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd  


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature= 1, contrast_mode='all',
                 base_temperature= 1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         print("contrast_feature:",contrast_feature,contrast_feature.shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
#         print("anchor_feature:",anchor_feature,anchor_feature.shape)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
#         print("anchor_dot_contrast:",anchor_dot_contrast,anchor_dot_contrast.shape)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         print("logits_max:",logits_max,logits_max.shape)
        logits = anchor_dot_contrast  - logits_max.detach()
#         print("logits:",logits,logits.shape)
        # tile mask
#         print("mask:",mask,mask.shape)
        mask = mask.repeat(anchor_count, contrast_count)
# #         print("mask_repeated:",mask,mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
#         print("logits_mask:",logits_mask,logits_mask.shape)
        mask = mask * logits_mask
#         print("mask_repeated:",mask,mask.shape)

        # compute log_prob
        exp_logits = torch.exp(logits)*logits_mask # a row coresponding to a sample with all the other samples
#         print("exp_logits:",exp_logits,exp_logits.shape)
        log_prob =   - torch.log(exp_logits.sum(1, keepdim=True))  #求一行中所有列的和
#         print("log_prob:",log_prob,log_prob.shape)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  #减去i对应位置的负对总和
#         print("mask.sum(1):",mask.sum(1),mask.sum(1).shape)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
#         print("loss:",loss,loss.shape)
        return loss        
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LDA')
    args= parser.parse_args()
    return args

if __name__ == '__main__':
    print("Training Starting!")
    args = parse_args()
    experiment_data = np.load("./ShuK_2024_08_03 16_52_data.npy",allow_pickle=True)
    # 获取当前时间
    now = datetime.now()
    # 格式化时间输出为"年-月-日 时:分"
    Exp_date = now.strftime("%Y_%m_%d %H_%M")
    feature = []
    EEG_All,EMG_All,Label_All  = get_rawdata(experiment_data)
    length,step = 200,100
    Train_data,train_label,Test_data,test_label = prepare_dataset(length,step,EEG_All,EMG_All,Label_All)
    Train_data,train_label = Feature_Engineering(Train_data,train_label)

    ##np.hstack((np.array(Train_data[0]).reshape(np.array(Train_data[0]).shape[0],-1),np.array(Train_data[1]).reshape(np.array(Train_data[0]).shape[0],-1)))
      
    # train_data = np.hstack((train_data,np.fft.fft(train_data,axis=2).real))
    # train_data = np.fft.fft(train_data,axis=2).imag
    # train_data_eeg = (train_data-train_data.mean(axis=0, keepdims=True))/train_data.std(axis=0, keepdims=True)
    # train_data = np.array(Train_data[1])#np.hstack((list(train_data[0]),list(train_data[1])))
    # train_data_emg = (train_data-train_data.mean(axis=0, keepdims=True))/train_data.std(axis=0, keepdims=True)
    # train_data = train_data_eeg#np.hstack((train_data_eeg,train_data_emg))#
    print("train_label.shape:",np.array(train_label).shape,np.unique(np.array(train_label), return_counts=True))

    # 加载模型
    # lda_loaded = load('./'+Exp_date+'_lda_model.joblib')
    Test_data,test_label = Feature_Engineering(Test_data,test_label)
    print("test_label.shape:",np.array(test_label).shape,np.unique(np.array(test_label), return_counts=True))

    feature.append({
                'train_data': Train_data,
                'train_label': train_label,
                'test_data': Test_data,
                'test_label': test_label,            
                })
    
    # np.save('D:/USTC/Program/metaBCI/MotorExecutionBCI/Features/'+
    #         'ShuK' +'_'+'2024_07_21 15_04_data'+'_features_400_100.npy',feature)

    # # 使用加载的模型进行预测（这里用测试集作为例子）#

    # # np.hstack((np.array(Test_data[0]).reshape(np.array(Test_data[0]).shape[0],-1),np.array(Test_data[1]).reshape(np.array(Test_data[0]).shape[0],-1)))
    # feature  = np.load('D:/USTC/Program/metaBCI/MotorExecutionBCI/Features/'+
    #         'ShuK' +'_'+'2024_07_21 15_04_data'+'_features_400_100.npy',allow_pickle=True)  
    Train_data,train_label,Test_data,test_label  = feature[0]['train_data'],feature[0]['train_label'],feature[0]['test_data'],feature[0]['test_label']
    train_data = np.array(Train_data[1])
    testdata =  np.array(Test_data[1]) # 
    # train_data = np.expand_dims(np.array(Train_data[0]),axis = 1)
    # testdata =  np.expand_dims(np.array(Test_data[0]),axis = 1) # 
    # testdata = np.hstack((testdata,np.fft.fft(testdata,axis=2).real))
    # testdata = np.fft.fft(testdata,axis=2).real  np.fft.fft(testdata,axis=2).imag,
    # train_data = np.array(Train_data[0])#np.hstack((list(train_data[0]),list(train_data[1])))
    # testdata_eeg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # testdata = np.array(Test_data[1])  
    # train_data = np.array(Train_data[1])#np.hstack((list(train_data[0]),list(train_data[1])))
    # testdata_emg = (testdata-train_data.mean(axis=0))/train_data.std(axis=0)
    # testdata = testdata_eeg#np.hstack((testdata_eeg,testdata_emg))#

    
    model_type = args.model  # 'MLP','LDA','SVM','RandomForest'  'Unimodal'
    print("model_type:",model_type)
    if model_type == 'LDA' or model_type == 'SVM' or model_type == 'RandomForest':
        Trained_model = train_PR_model(train_data,train_label,Exp_date,type=model_type)
        y_pred = Trained_model.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
        accuracy_loaded = accuracy_score(test_label, y_pred)
        cm = confusion_matrix(test_label, y_pred)

    elif model_type == 'Late_Fusion':
        train_data = Train_data
        testdata =  Test_data # 
        print("Test_data:",type(Test_data[0]),type(Test_data[1]))

        Trained_model = train_PR_model(train_data,train_label,Exp_date,type=model_type)
        test_dataset = TensorDataset(torch.tensor(testdata[0],dtype=torch.float),
                                    torch.tensor(testdata[1],dtype=torch.float),
                                    torch.tensor(np.array(test_label),dtype=torch.long))
        test_loader1 = DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=False)
        y_pred, Test_Label = test_Fusion(Trained_model,test_loader1)  
        accuracy_loaded = accuracy_score(test_label, y_pred)
        cm = confusion_matrix(test_label, y_pred)
    else:
        Trained_model = train_PR_model(train_data,train_label,Exp_date,type=model_type)
        test_dataset = TensorDataset(torch.tensor(testdata,dtype=torch.float),
                                        torch.tensor(np.array(test_label),dtype=torch.long))
        test_loader1 = DataLoader(dataset=test_dataset,
                            batch_size=256,
                            shuffle=False)
        y_pred, test_Label = test_Deeplearning(Trained_model,test_loader1)  
        accuracy_loaded = accuracy_score(test_label, y_pred)
        cm = confusion_matrix(test_label, y_pred)
    print(f"Accuracy with {model_type} model: {accuracy_loaded:.4f}")    
    print("confusion_matrix:",'\n',cm)
    # 假设lda是已经训练好的LDA模型
    dump(Trained_model, './'+Exp_date+'_{}_model.joblib'.format(model_type))
    del Trained_model

    # Trained_model = train_PR_model(train_data,train_label,Exp_date,type='SVM')
    # y_pred = Trained_model.predict(np.reshape(testdata[:,:],(testdata.shape[0],-1)))
    # accuracy_loaded = accuracy_score(test_label, y_pred)
    # cm = confusion_matrix(test_label, y_pred)
    # print(f"Accuracy with SVM model: {accuracy_loaded:.4f}")    
    # print("confusion_matrix:",'\n',cm)
    # # 假设svm是已经训练好的SVM模型
    # # dump(svm, './'+Exp_date+'_svm_model.joblib')
    # del Trained_model



    # Trained_model = train_PR_model(train_data,train_label,Exp_date,type='MLP')
    # test_dataset = TensorDataset(torch.tensor(testdata,dtype=torch.float),
    #                                    torch.tensor(np.array(test_label),dtype=torch.long))
    # test_loader1 = DataLoader(dataset=test_dataset,
    #                         batch_size=256,
    #                         shuffle=False)
    # y_pred, Test_Label = test_Deeplearning(Trained_model,test_loader1)  
    # accuracy_loaded = accuracy_score(test_label, y_pred)
    # cm = confusion_matrix(test_label, y_pred)
    # print(f"Accuracy with MLP model: {accuracy_loaded:.4f}")    
    # print("confusion_matrix:",'\n',cm)   
    # # 假设MLP是已经训练好的MLP模型
    # # dump(MLP, './'+Exp_date+'_MLP_model.joblib')
    # del Trained_model

    # Trained_model = train_PR_model(train_data,train_label,Exp_date,type='Unimodal')
    # test_dataset = TensorDataset(torch.tensor(testdata,dtype=torch.float),
    #                                    torch.tensor(np.array(test_label),dtype=torch.long))
    # test_loader1 = DataLoader(dataset=test_dataset,
    #                         batch_size=256,
    #                         shuffle=False)
    # y_pred, Test_Label = test_Deeplearning(Trained_model,test_loader1)  
    # accuracy_loaded = accuracy_score(test_label, y_pred)
    # cm = confusion_matrix(test_label, y_pred)
    # print(f"Accuracy with Unimodal model: {accuracy_loaded:.4f}")    
    # print("confusion_matrix:",'\n',cm)
    # # 假设Unimodal是已经训练好的Unimodal模型
    # # dump(Unimodal, './'+Exp_date+'_Unimodal_model.joblib')
    # del Trained_model
 
    # train_data = Train_data
    # testdata =  Test_data # 
    # Trained_model = train_PR_model(train_data,train_label,Exp_date,type='Late_Fusion')
    # test_dataset = TensorDataset(torch.tensor(testdata[0],dtype=torch.float),
    #                              torch.tensor(testdata[1],dtype=torch.float),
    #                                    torch.tensor(np.array(test_label),dtype=torch.long))
    # test_loader1 = DataLoader(dataset=test_dataset,
    #                         batch_size=256,
    #                         shuffle=False)
    # y_pred, Test_Label = test_Fusion(Trained_model,test_loader1)  
    # accuracy_loaded = accuracy_score(test_label, y_pred)
    # cm = confusion_matrix(test_label, y_pred)
    # 假设Unimodal是已经训练好的Unimodal模型
    # dump(Trained_model, './'+Exp_date+'_LateFusion_model.joblib')
    # print(f"Accuracy with Late_Fusion model: {accuracy_loaded:.4f}")    
    # print("confusion_matrix:",'\n',cm)
