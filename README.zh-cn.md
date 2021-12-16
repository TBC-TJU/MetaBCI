<!-- PROJECT LOGO -->
[English](README.md)
<br />
<p align="center">
  <!-- <a href="https://github.com/Mrswolf/brainda">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Brainda</h3>

  <p align="center">
    脑-机接口数据和算法库
    <br />
    <a href="https://brainda.readthedocs.io/en/latest"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/Mrswolf/brainda">View Demo</a> -->
    ·
    <a href="https://github.com/Mrswolf/brainda/issues">Report Bug</a>
    ·
    <a href="https://github.com/Mrswolf/brainda/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">目录</h2></summary>
  <ol>
    <li>
      <a href="#关于本项目">关于本项目</a>
      <ul>
        <li><a href="#项目主要特点">项目主要特点</a></li>
      </ul>
    </li>
    <li>
      <a href="#安装">安装</a>
    </li>
    <li>
        <a href="#使用">使用</a>
        <ul>
            <li><a href="#加载数据">加载数据</a></li>
            <li><a href="#预处理">预处理</a></li>
            <li><a href="#机器学习流程">机器学习流程</a></li>
        </ul>
    </li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#参与项目">参与项目</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 关于本项目

当我还是脑-机接口小白的时候，有三件事最让我心烦：
1. 打导电膏
2. 预处理各种不同格式的脑电数据
3. 一遍又一遍的在MATLAB下复制粘贴算法程序

对于第一个问题，我觉得是没救了（也许十年内有希望替换注射器？）。其他的问题我则在Python平台下找到了答案。对于第二个问题，当我开始学习Python和MNE时，我开始尝试用一套统一的接口来简化数据的获取和预处理，直到后来发现[MOABB](https://github.com/NeuroTechX/moabb)发布了。MOABB的API接口设计和内部代码显然要比我自己搭建的简陋框架高级的多，于是我开始利用MOABB获取数据。对于第三个问题，Scikit-learn的fit和transform提供了一种优雅的机器学习抽象，使我能最大程度的复用已有的代码。通过综合MOABB和Scikit-learn设计的优点，我创建本项目作为我日常研究用脚手架，主要集中在创建数据集和复现算法。


### 项目主要特点
1. 对MOABB数据获取API的改进
   - 添加钩子函数以便更灵活的控制预处理流程
   - 利用joblib加速多人数据的载入和处理
   - 为部分地区获取数据添加代理选项
   - 在数据Meta中添加更多有用信息
   - 其他小的改进
  
2. 现有的脑-机接口算法，均为Python实现
    - 分解算法
      - MI的CSP, MultiCSP, FBCSP
      - SSVEP的ExtendCCA,TRCA, EnsembleTRCA, SSCOR
    - 流形学习
      - 基本黎曼几何操作
      - 对齐算法
      - 黎曼普氏分析
    - 深度学习
      - EEGNet
    - 迁移学习
       - MEKT


<!-- GETTING STARTED -->
## 安装

1. 克隆项目到本地
   ```sh
   git clone https://github.com/Mrswolf/brainda.git
   ```
2. 切换到brainda目录下
   ```sh
   cd brainda
   ```
3. 安装依赖项
   ```sh
   pip install -r requirements.txt 
   ```
4. 在编辑模式下安装brainda
   ```sh
   pip install -e .
   ```

<!-- USAGE EXAMPLES -->
## 使用

### 加载数据

最简单的情况是使用数据集作者的默认设置加载数据：
```python
from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery

dataset = AlexMI() # declare the dataset
paradigm = MotorImagery(
    channels=None, 
    events=None,
    intervals=None,
    srate=None
) # declare the paradigm, use recommended Options

print(dataset) # see basic dataset information

# X,y are numpy array and meta is pandas dataFrame
X, y, meta = paradigm.get_data(
    dataset, 
    subjects=dataset.subjects, 
    return_concat=True, 
    n_jobs=-1, 
    verbose=False)
print(X.shape)
print(meta)
```
如果本地还没有数据集，程序会自动下载数据到本地，通常储存在`~/mne_data`文件夹下。不过，你也可以提前下载数据，指定储存目录：
```python
dataset.download_all(
    path='/your/datastore/folder', # save folder
    force_update=False, # re-download even if the data exist
    proxies=None, # add proxy if you need, the same as the Request package
    verbose=None
)

# If you encounter network connection issues, try this
# dataset.download_all(
#     path='/your/datastore/folder', # save folder
#     force_update=False, # re-download even if the data exist
#     proxies={
#         'http': 'socks5://user:pass@host:port',
#         'https': 'socks5://user:pass@host:port'
#     },
#     verbose=None
# )

```
也可以指定导联、事件、试次长度、采样率和被试等选项：
```python
paradigm = MotorImagery(
    channels=['C3', 'CZ', 'C4'], 
    events=['right_hand', 'feet'],
    intervals=[(0, 2)], # 2 seconds
    srate=128
)

X, y, meta = paradigm.get_data(
    dataset, 
    subjects=[2, 4], 
    return_concat=True, 
    n_jobs=None, 
    verbose=False)
print(X.shape)
print(meta)
```
或者为不同事件指定不同的试次长度。这种情况下，X，y和meta都是字典结构：
```python
dataset = AlexMI()
paradigm = MotorImagery(
    channels=['C3', 'CZ', 'C4'], 
    events=['right_hand', 'feet'],
    intervals=[(0, 2), (0, 1)], # 2s for right_hand, 1s for feet
    srate=128
)

X, y, meta = paradigm.get_data(
    dataset, 
    subjects=[2, 4], 
    return_concat=False, 
    n_jobs=-1, 
    verbose=False)
print(X['right_hand'].shape, X['feet'].shape)
```
### 预处理
paradigm.get_data`的处理流程如下：

<p align="center">
    <img src="images/get_data_flow.jpg" width="700" height="150">
</p>

brainda提供3个钩子函数用来更加灵活的控制预处理流程，基本逻辑同MNE的处理流程完全一致：

```python
dataset = AlexMI()
paradigm = MotorImagery()

# add 6-30Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, 
        l_trans_bandwidth=2, 
        h_trans_bandwidth=5, 
        phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

def epochs_hook(epochs, caches):
    # do something with epochs object
    print(epochs.event_id)
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches

def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
    print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches

paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

X, y, meta = paradigm.get_data(
    dataset, 
    subjects=[1], 
    return_concat=True, 
    n_jobs=-1, 
    verbose=False)
```
如果数据集作者提供了这些钩子，brainda会隐式的调用这些钩子。当然，你总是可以替换掉它们。

### 机器学习流程

这是一个简单的2分类CSP算法，展示如何使用brainda和Scikit-learn分类数据。

```python
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import CSP

dataset = AlexMI()
paradigm = MotorImagery(events=['right_hand', 'feet'])

# add 6-30Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5, phase='zero-double', verbose=False)
    return raw, caches

paradigm.register_raw_hook(raw_hook)

X, y, meta = paradigm.get_data(
    dataset, 
    subjects=[3], 
    return_concat=True, 
    n_jobs=-1, 
    verbose=False)

# 5-fold cross validation
set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# CSP with SVC classifier
estimator = make_pipeline(*[
    CSP(n_components=4),
    SVC()
])

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels==y[test_ind]))
print(np.mean(accs))
```
如果一切正常，应该会得到正确率0.925。

<!-- _For more examples, please refer to the [Documentation](https://github.com/Mrswolf/brainda)_ -->

<!-- ROADMAP -->
## 路线图
- add demos
- add documents
- more datasets for P300
- more BCI algorithms
  
See the [open issues](https://github.com/Mrswolf/brainda/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## 参与项目

任何贡献都是欢迎的，尤其欢迎各位大佬提交Python版的算法。

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

My Email: swolfforever@gmail.com

Project Link: [https://github.com/Mrswolf/brainda](https://github.com/Mrswolf/brainda)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Mrswolf/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/Mrswolf/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Mrswolf/repo.svg?style=for-the-badge
[forks-url]: https://github.com/Mrswolf/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/Mrswolf/repo.svg?style=for-the-badge
[stars-url]: https://github.com/Mrswolf/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/Mrswolf/repo.svg?style=for-the-badge
[issues-url]: https://github.com/Mrswolf/repo/issues
[license-shield]: https://img.shields.io/github/license/Mrswolf/repo.svg?style=for-the-badge
[license-url]: https://github.com/Mrswolf/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/Mrswolf
