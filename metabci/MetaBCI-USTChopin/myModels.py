
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from functools import partial
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.signal import stft

class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2*2, bias=False,dilation=2)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False,dilation=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
#         print('SA x:',x.shape)
        x = self.conv1(x)
        
        return self.sigmoid(x)

def get_stft(data,fs=500,nperseg=50, noverlap=40,nfft=100):
    data_np = data.cpu().numpy()
    # 对每个通道进行STFT变换
    stft_results = []
    for i in range(data_np.shape[0]):  # 遍历样本
        # 初始化一个空列表来存储每个通道的STFT结果 
        channel_stft = []
        for j in range(data_np.shape[-1]):  # 遍历通道
            # 对单个样本的信号进行STFT变换
            freq, time, Zxx = stft(data_np[i, 0, :, j],fs=500,nperseg=50, noverlap=40,nfft=100)
            # 将STFT结果添加到列表
            channel_stft.append(np.abs(Zxx))  # 只保留幅度信息
        # 将列表转换回NumPy数组，并计算所有样本的STFT结果
        stft_results.append(np.stack(channel_stft))

    # 将STFT结果转换为PyTorch张量
    stft_tensors = torch.stack([torch.from_numpy(result) for result in stft_results])
    return stft_tensors.cuda()

class MLP_Test(torch.nn.Module):
    def __init__(self,dropout,input_size,hiden_size,channel,classes):
        super(MLP_Test,self).__init__()
        self.channel = channel
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(channel*input_size,hiden_size),
            torch.nn.BatchNorm1d(hiden_size,affine =True),
            # torch.nn.LayerNorm(300),
            torch.nn.ReLU(inplace=True),        
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiden_size,hiden_size//2),
            torch.nn.BatchNorm1d(hiden_size//2,affine =True),
            # torch.nn.LayerNorm(100),
            torch.nn.ReLU(inplace=True),  
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiden_size//2,50),
            torch.nn.BatchNorm1d(50,affine =True),            
            # torch.nn.LayerNorm(50),
            torch.nn.ReLU(inplace=True),   
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(50,3)
        )   
        self.fc2 = torch.nn.Linear(50,classes)
        # self.fc3 = torch.nn.Linear(50,3)

    def forward(self, data):
        x = data
        x2 = self.mlp1(x.reshape(x.size(0),-1))  
        # x2= self.mlp2(k_fft_abs.reshape(x.size(0),-1))    
        # x = self.mix(torch.cat((x1,x2),dim=1).reshape(x.size(0),-1))    
        # x_c = self.fc1(x2)
        x_r = self.fc2(x2)
        # x_w = self.fc3(x2)
        return x_r

class EEG_model(torch.nn.Module):
    def __init__(self,dropout,input_size,hiden_size,channel,classes):
        super(EEG_model,self).__init__()
        self.channel = channel
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=7,out_channels=7,kernel_size=[30,1],stride=1,padding=0,groups=7),  
            torch.nn.BatchNorm2d(7,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            # torch.nn.ReLU(inplace=True)          
        )
        
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(7,hiden_size),
            torch.nn.BatchNorm1d(hiden_size,affine =True),
            # torch.nn.LayerNorm(300),
            torch.nn.ReLU(inplace=True),        
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiden_size,hiden_size//2),
            torch.nn.BatchNorm1d(hiden_size//2,affine =True),
            # torch.nn.LayerNorm(100),
            torch.nn.ReLU(inplace=True),  
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiden_size//2,50),
            torch.nn.BatchNorm1d(50,affine =True),            
            # torch.nn.LayerNorm(50),
            torch.nn.ReLU(inplace=True),   
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(50,3)
        )   
        self.fc2 = torch.nn.Linear(50,classes)
        # self.fc3 = torch.nn.Linear(50,3)

    def forward(self, data):
        x = self.conv1(data)
        x = torch.var(x,dim=3)
        x = torch.log(torch.abs(x))
        x2 = self.mlp1(x.reshape(x.size(0),-1))  
        # x2= self.mlp2(k_fft_abs.reshape(x.size(0),-1))    
        # x = self.mix(torch.cat((x1,x2),dim=1).reshape(x.size(0),-1))    
        # x_c = self.fc1(x2)
        x_r = self.fc2(x2)
        # x_w = self.fc3(x2)
        return x_r


class CNN_LSTM(torch.nn.Module):
    def __init__(self,dropout,input_dim,channel,channel_kin):
        super(CNN_LSTM,self).__init__()
        self.channel = channel
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim,out_channels=64,kernel_size=15,stride=1,padding=7),
            # torch.nn.BatchNorm1d(64,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            # torch.nn.ReLU(inplace=True)          
        )
        self.maxpooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(64,32,7,1,3),
            # torch.nn.BatchNorm1d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            # torch.nn.ReLU(inplace=True)          
        )
        self.maxpooling2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.hidden_size = 50
        self.num_layers = 1
        self.lstm = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)  #(input_size, hidden_size, num_layers)
        
        self.fc2 = torch.nn.Linear(50*1,channel_kin)
        
        
    def forward(self, data,label):

        # x = data[:,:,:,:18]
        if self.channel == 18:
            x = data[:,:,:,:18]
        else:
            x = data[:,:,:,18:23]
        # k_fft = torch.fft.rfft(x.permute(0, 1, 3, 2).contiguous(), dim=-1)
        # k_fft_abs = torch.abs(k_fft)
        x = x.reshape(x.size(0),x.size(2),x.size(3))
        # x = x.permute(0,2,1)

        x1 = self.conv0(x)  
        x11 = self.maxpooling1(x1)
        # print("x11:",x11.shape)
        x2 = self.conv1(x11)  
        x21 = self.maxpooling2(x2)
        # print("x21:",x21.shape)
        x30 = x21.view(x21.size(0),1,-1)
        x31 = x30.permute(0,2,1)
        # print("x30.shape:",x31.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        

        out, hidden = self.lstm(x31 ,(h0,c0))  # (batch_size, sequence_length, input_size)  
        # print("x3:",x3)
        # print(x3.size())

        # x2= self.mlp2(k_fft_abs.reshape(x.size(0),-1))    
        # x = self.mix(torch.cat((x1,x2),dim=1).reshape(x.size(0),-1))    
        # x_c = self.fc1(x2)
        # print("out:",out.shape)
        x_r = self.fc2(out[:,-1,:].reshape(x.size(0),-1))
        # x_w = self.fc3(x2)
        # print("x_r:",x_r)
        return x_r,x_r,x_r

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始化 LSTM 的隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # LSTM 输出的 out 是最后一个时间步的隐藏状态序列
        
        # 取最后一个时间步的隐藏状态作为分类输入
        out = self.fc(out[:, -1, :])
        return out


class Early_fusion(torch.nn.Module):
    def __init__(self,dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes):
        super(Early_fusion,self).__init__()
 
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(channel_eeg,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )
        self.emg_conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(channel_emg,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),  
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=13,stride=1,padding=6),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(32*3,3,[4,3],[4,1],[0,1]),
            torch.nn.BatchNorm2d(3),
            # torch.nn.LayerNorm([32,25,9]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3*int(input_dim1/4*23),512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )        
        self.mlp3 = torch.nn.Linear(512,classes)
    def forward(self, data1,data2):
        
        x1 = self.conv0(data1) 
        
        a1 = self.emg_conv0(data2)

        xa = torch.cat((x1,a1),dim=3)
        xa1 = self.conv1(xa)

        xa2 = self.conv2(xa1+xa)

        xa3 = self.conv3(xa2+xa1)

        f4 = self.conv4(torch.cat((xa1,xa2,xa3),dim=1))

        f1 = self.mlp1(f4.view(f4.size(0),-1))  #tensor to 1 Dimension
        f5 = self.mlp2(f1)
        f = self.mlp3(f5)

        return f,f5


class Late_Fusion(torch.nn.Module):
    def __init__(self,dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes):
        super(Late_Fusion,self).__init__()
 
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(channel_eeg,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),  
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=13,stride=1,padding=6),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
   
        self.emg_conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(channel_emg,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )
        self.emg_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0),  
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,5]),
            torch.nn.ReLU(inplace=True)          
        )
        self.emg_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,5]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.emg_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,5]),
            torch.nn.ReLU(inplace=True)          
        )  
 
        self.conv4 = torch.nn.Sequential(
            # torch.nn.Conv2d(32*3,3,[4,3],[4,1],[0,1]),
            torch.nn.Conv2d(32*3,3,[3,4],[1,4],[1,0]),
            # torch.nn.Conv2d(32*3,3,3,1,1),
            torch.nn.BatchNorm2d(3),
            # torch.nn.LayerNorm([32,25,9]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3*int(input_dim1/4),512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )        
  
        self.emg_conv4 = torch.nn.Sequential(
            # torch.nn.Conv2d(32*3,3,[3,4],[1,4],[1,0]),
            torch.nn.Conv2d(32*3,3,3,1,1),
            torch.nn.BatchNorm2d(3),
            # torch.nn.LayerNorm([32,25,5]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.emg_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3*int(input_dim2),512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )        
        self.emg_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )        
       
        # self.mlp3 = torch.nn.Linear(512,classes)                
        # self.mlp4 = torch.nn.Linear(512,classes)      
        self.mlp5 = torch.nn.Linear(512,classes)      
    
    def forward(self, data1,data2):
        x1 = self.conv0(data1) 
        x2 = self.conv1(x1) 
        
        a1 = self.emg_conv0(data2)
        a2 = self.emg_conv1(a1)
    
 
        x3 = self.conv2(x2 + x1)# 
        a3 = self.emg_conv2(a2+a1)
 
        x = self.conv3(x3 +x2 )#+ 
        a = self.emg_conv3(a3+a2)

 
        x = torch.cat((x2,x3,x),dim=1)
        x4 = self.conv4(x )

        a = torch.cat((a2,a3,a),dim=1)
        a4 = self.emg_conv4(a )  
 
        x = self.mlp1(x4.view(x4.size(0),-1))  #tensor to 1 Dimension
        x5 = self.mlp2(x)
 
        a = self.emg_mlp1(a4.view(a4.size(0),-1))  #tensor to 1 Dimension
        a5 = self.emg_mlp2(a)

        f5 = 0.7*x5+0.3*a5 #torch.cat((x5,a5),dim=1)
        # f1 = self.mlp3(a5)
        # f2 = self.mlp4(x5)
        f3 = self.mlp5(f5)
        return f3,x5,a5 #torch.max(f1,f2)f1+f2+



class EEG_model2(torch.nn.Module):
    def __init__(self,dropout,input_dim1,channel_eeg,classes):
        super(EEG_model2,self).__init__()
 
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(channel_eeg,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),  
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=13,stride=1,padding=6),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )

        self.conv4 = torch.nn.Sequential(
            # torch.nn.Conv2d(32*3,3,[4,3],[4,1],[0,1]),
            # torch.nn.Conv2d(32*3,3,[3,4],[1,4],[1,0]),
            torch.nn.Conv2d(32*3,3,3,1,1),
            torch.nn.BatchNorm2d(3),
            # torch.nn.LayerNorm([32,25,9]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3*int(input_dim1),512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )             
        # self.mlp3 = torch.nn.Linear(512,classes)                
        self.mlp4 = torch.nn.Linear(512,classes)      
    
    
    def forward(self, data1):
        x1 = self.conv0(data1) 
        x2 = self.conv1(x1) 
        x3 = self.conv2(x2 + x1)#  
        x = self.conv3(x3 +x2 )#+ 
        x = torch.cat((x2,x3,x),dim=1)
        x4 = self.conv4(x )
        x = self.mlp1(x4.view(x4.size(0),-1))  #tensor to 1 Dimension
        x5 = self.mlp2(x) 
        f2 = self.mlp4(x5)
 
        return f2
    
class Unimodal(torch.nn.Module):
    def __init__(self,dropout,input_dim,channel,classes):
        super(Unimodal,self).__init__()
        self.channel = channel
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv01 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),

        )

        self.conv02 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8), 
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )

        self.conv03 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )

        self.conv07 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )

        self.conv04 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )

        self.conv05 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )

        self.conv06 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,1,1,0),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8,8,3,1,1),
            torch.nn.BatchNorm2d(8),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=56,out_channels=24,kernel_size=3,stride=1,padding=1),  
            torch.nn.BatchNorm2d(24,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=24,out_channels=24,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(24,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=24,out_channels=24,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(24,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )

        self.conv4 = torch.nn.Sequential(
            # torch.nn.Conv2d(32*3,32,[4,3],[4,1],[0,1]),
            torch.nn.Conv2d(24*3,3,3,1,1),
            torch.nn.BatchNorm2d(3),
            # torch.nn.LayerNorm([32,25,9]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3*input_dim,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )        
        self.mlp3 = torch.nn.Linear(512,classes)
    def forward(self, data):
        x01 = self.conv01(torch.unsqueeze(data[:,0,:,:],dim=1))
        x02 = self.conv02(torch.unsqueeze(data[:,1,:,:],dim=1))
        x03 = self.conv03(torch.unsqueeze(data[:,2,:,:],dim=1))
        x04 = self.conv01(torch.unsqueeze(data[:,3,:,:],dim=1))
        x05 = self.conv02(torch.unsqueeze(data[:,4,:,:],dim=1))
        x06 = self.conv03(torch.unsqueeze(data[:,5,:,:],dim=1))
        x07 = self.conv03(torch.unsqueeze(data[:,6,:,:],dim=1))

        x = torch.cat((x01,x02,x03,x04,x05,x06,x07),dim=1)
        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x3 = self.conv3(x1+x2)

        f4 = self.conv4(torch.cat((x1,x2,x3),dim=1))

        f1 = self.mlp1(f4.view(f4.size(0),-1))  #tensor to 1 Dimension
        f5 = self.mlp2(f1)
        f = self.mlp3(f5)

        return f

class Uni_FFT_modal(torch.nn.Module):
    def __init__(self,dropout,input_dim,channel,channel_kin):
        super(Uni_FFT_modal,self).__init__()
        self.channel = channel
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(18,32,1,1,0),
            torch.nn.BatchNorm2d(32),
        )

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),  
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        ) 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=13,stride=1,padding=6),
            torch.nn.BatchNorm2d(32,affine =True),
            # torch.nn.LayerNorm([32,100,18]),
            torch.nn.ReLU(inplace=True)          
        )

        self.conv4 = torch.nn.Sequential(
            # torch.nn.Conv2d(32*3,32,[4,3],[4,1],[0,1]),
            torch.nn.Conv2d(32*3,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            # torch.nn.LayerNorm([32,25,9]),     
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(
            # torch.nn.Linear(32*int(input_dim/4*self.channel),512),
            torch.nn.Linear(32*51*11,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(512,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )        
        self.mlp3 = torch.nn.Linear(512,channel_kin)
    def forward(self, data,label):
        if self.channel == 18:
            # x_fft = torch.fft.fft(data[:,:,:,:18], 100, dim=2).real
            # x_fft_imag = torch.fft.fft(data[:,:,:,:18], 100, dim=2).imag
            eeg_stft = get_stft(data[:,:,:,:18],fs=500,nperseg=50, noverlap=40,nfft=100) #B,18,51,11

            # print("pre  x_fft:",x_fft.shape)

            # x_fft = x_fft.unsqueeze(1)
            # print("x_fft:",x_fft.shape)
            x = self.conv0(eeg_stft) 
            # x = self.conv0(torch.cat((data[:,:,:,:18],x_fft),dim=1)) 
            # x = self.conv0(x_fft_imag) 

        elif self.channel == 5:
            x_fft = torch.fft.fft(data[:,:,:,18:23], 100, dim=2).real
            # x_fft_imag = torch.fft.fft(data[:,:,:,18:23], 100, dim=2).imag

            # print("pre  x_fft:",x_fft.shape)

            # x_fft = x_fft.unsqueeze(1)
            # print("x_fft:",x_fft.shape)
            x = self.conv0(torch.cat((data[:,:,:,18:23],x_fft),dim=1)) 
 
            # x = self.conv0(x_fft_imag) 

        x1 = self.conv1(x)

        x2 = self.conv2(x1+x)

        x3 = self.conv3(x2+x1)

        f4 = self.conv4(torch.cat((x1,x2,x3),dim=1))

        f1 = self.mlp1(f4.view(f4.size(0),-1))  #tensor to 1 Dimension
        f5 = self.mlp2(f1)
        f = self.mlp3(f5)

        return f,f5
