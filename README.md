# MetaBCI

## Welcome! 
本项目基于开源MetaBCI平台，搭建了一套实现全链条技术国产化的沉浸式VR康复训练系统。通过国产高性能3D引擎构建高真实感的虚拟康复场景，实时呈现康复任务，同步反馈运动想象解码结果，提升患者训练的专注程度；集成博睿康脑电采集系统和MetaBCI平台内自带的FBCSP算法，实现了脑电信号的实时传输与识别，指令识别准确率达82.5%。系统根据解码指令驱动电刺激设备，触发患者特定手部肌肉分级收缩（强度可调），有效增强训练专注度与康复效果。从3D场景渲染、脑电信号采集与解码到电刺激执行，全链路采用国产化技术装备，真正构建了“感知-计算-干预”一体化的全国产沉浸式VR康复系统。

## Content

- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [What are we doing?](#what-are-we-doing)
  - [Section on Updates and Fixes](#Section on Updates and Fixes)
  - [Usage Instructions](#Usage Instructions)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

* 依托MetaBCI开源平台架构，本项目高效集成VR刺激反馈界面、博睿康
脑电采集系统通信接口、运动想象（MI）实时解码算法以及自研电刺激设备控制模块，实现结合VR与电刺激技术的运动障碍患者康复训练系统。
  - 刺激呈现：实现上位机与VR设备的实时通信。VR眼镜呈现康复任务界面，并实时反馈MI解码结果，增强沉浸式训练体验。
  - 数据获取：在brainflow子平台改进博睿康数据解析模块，实时采集并传输脑电信号至处理终端。
  - 信号处理：调用brainda子平台的FBCSP算法，对脑电信号进行实时特征提取与模式识别，解码用户运动意图为控制指令。
  - 外设控制：在brainflow子平台新增电刺激设备控制模块，基于解码得到的指令控制电刺激设备，触发患者特定手部肌肉收缩，实现闭环康复训练目的。


## Section on Updates and Fixes

* Updates
   - 新增电刺激控制模块    Brainflow	  metabci\brainflow\ ElectroStimulator.py    1.ElectroStimulator()
   - 新增刺激标签传输功能    Brainflow	  metabci\brainflow\amplifiers.py    1.BaseAmplifier()  2.Marker()

* Fixes
   - 优化Neuracle放大器数据流模块  Brainflow    metabci\brainflow\amplifiers.py    1.Neuracle()
     
##  Usage Instructions

  - 在刺激电脑A上首先运行Blank_stim.py文件，然后打开VR刺激上位机程序TunerlRehabilitation.exe；
  - 在脑电接收识别电脑B上打开博睿康采集程序，点击数据转发，运行MIprocess_Online.py文件；
  - 设置电脑A的康复方式，依次点击模式选择、开始复健、指定任务复健，向左转向和向右转向都设置为10次，点击开始复健。



## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China


## Contact

Email: 1364747481@qq.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
