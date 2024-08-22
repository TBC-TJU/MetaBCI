import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from functools import partial
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.signal import stft
from .base import SkorchNet
from collections import OrderedDict

@SkorchNet
class Late_Fusion(torch.nn.Module):
    def __init__(self,dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes):
        super().__init__()
 
#         self.params = nn.Parameter(torch.randn([num_classes, 512]))
        self.conv0 = torch.nn.Sequential(OrderedDict(
            [
            
                    ("conv",nn.Conv2d(channel_eeg,32,1,1,0)),
                    ("bn", nn.BatchNorm2d(32)),
                    
                
        ]))
        
        self.conv1 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,3,1,1)),
                ("bn", nn.BatchNorm2d(32)),
                ("relu", nn.ReLU(inplace=True)),
            ]) 
        )
        self.conv2 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,7,1,3)),
                ("bn", nn.BatchNorm2d(32)),
                ("relu", nn.ReLU(inplace=True)),
            ])    
        ) 
        self.conv3 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,13,1,6)),
                ("bn", nn.BatchNorm2d(32)),
                ("relu", nn.ReLU(inplace=True)),
            ])    
        )  
           
        self.emg_conv0 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(channel_emg,32,1,1,0)),  
                ("bn", nn.BatchNorm2d(32)),
               
            ])
        )
        self.emg_conv1 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,1,1,0)),  
                ("bn", nn.BatchNorm2d(32,affine =True)),
                ("relu", nn.ReLU(inplace=True))
            ])
        )
        self.emg_conv2 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,3,1,1)),
                ("bn", nn.BatchNorm2d(32,affine =True)),
                ("relu", nn.ReLU(inplace=True))
            ])
        ) 
                
         
        self.emg_conv3 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32,32,5,1,2)),
                ("bn", nn.BatchNorm2d(32,affine =True)),
                ("relu", nn.ReLU(inplace=True))
            ])
                  
        )  
 
        self.conv4 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32*3,3,[3,4],[1,4],[1,0])),
                ("bn", nn.BatchNorm2d(3)),
                ("relu", nn.ReLU(inplace=True)),
                ("dropout", nn.Dropout(dropout)),
            ])
            # # torch.nn.Conv2d(32*3,3,[4,3],[4,1],[0,1]),
            # torch.nn.Conv2d(32*3,3,[3,4],[1,4],[1,0]),
            # # torch.nn.Conv2d(32*3,3,3,1,1),
            # torch.nn.BatchNorm2d(3),
            # # torch.nn.LayerNorm([32,25,9]),     
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
        )
        self.mlp1 = torch.nn.Sequential(OrderedDict(
            [
                ("linear", nn.Linear(3*int(input_dim1/4),512)),
                ("bn", nn.BatchNorm1d(512)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(dropout)),
            ])
            # torch.nn.Linear(3*int(input_dim1/4),512),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout)
        )
        
        self.mlp2 = torch.nn.Sequential(OrderedDict(
            [
                ("linear", nn.Linear(512,512)),
                ("bn", nn.BatchNorm1d(512)),
                ("relu", nn.ReLU()),
            ])
            # torch.nn.Linear(512,512),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU()
        )        
  
        self.emg_conv4 = torch.nn.Sequential(OrderedDict(
            [
                ("conv", nn.Conv2d(32*3,3,3,1,1)),
                ("bn", nn.BatchNorm2d(3)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(dropout)),
            ])
            # torch.nn.Conv2d(32*3,3,[3,4],[1,4],[1,0]),
            # torch.nn.Conv2d(32*3,3,3,1,1),
            # torch.nn.BatchNorm2d(3),
            # # torch.nn.LayerNorm([32,25,5]),     
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
        )
        self.emg_mlp1 = torch.nn.Sequential(OrderedDict(
            [
                ("linear", nn.Linear(3*int(input_dim2),512)),
                ("bn", nn.BatchNorm1d(512)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(dropout)),
            ])
            # torch.nn.Linear(3*int(input_dim2),512),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout)
        )        
        self.emg_mlp2 = torch.nn.Sequential(OrderedDict(
            [
                ("linear", nn.Linear(512,512)),
                ("bn", nn.BatchNorm1d(512)),
                ("relu", nn.ReLU()),
            ])
            # torch.nn.Linear(512,512),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU()
        )        
       
        # self.mlp3 = torch.nn.Linear(512,classes)                
        # self.mlp4 = torch.nn.Linear(512,classes)      
        self.mlp5 = torch.nn.Linear(512,classes)      
        # self.model_eeg = nn.Sequential(self.conv0, self.conv1, self.conv2, self.fc_layer)
        # self.model_emg = nn.Sequential(self.emg_conv0, self.emg_conv1, self.emg_conv2, self.fc_layer)
    def forward(self, data):
        data1, data2 = data
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
        return f3 #torch.max(f1,f2)f1+f2+


class Late_Fusion_Attention(torch.nn.Module):
    def __init__(self,dropout,input_dim1,channel_eeg,input_dim2,channel_emg,classes,classes_attention):
        super(Late_Fusion_Attention,self).__init__()
 
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
        self.mlp4 = torch.nn.Linear(512,classes_attention)      
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
        f2 = self.mlp4(x5)
        f3 = self.mlp5(f5)
        return f3,f2,a5 #torch.max(f1,f2)f1+f2+


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