# Brainda

## Welcome!

First and foremost, Welcome!

Thank you for visiting the Brainda repository which was initially released at [this repo](https://github.com/Mrswolf/brainda) and reorganized here. This project is meant to provide datasets and decoding algorithms for BCI research, using python, as a part of the MetaBCI project which aims to provide a python platform for BCI users to design paradigm, collect data, process signals, present feedbacks and drive robots.

This document is a hub to give you some information about the project. Jump straight to one of the sections below, or just scroll down to find out more.

- [Brainda](#brainda)
  - [Welcome!](#welcome)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Loading](#data-loading)
    - [Preprocessing](#preprocessing)
    - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* BCI datasets come in different formats and standards
* It's tedious to figure out the details of the data
* Lack of python implementations of modern decoding algorithms

If someone new to the BCI wants to do some interesting research, most of their time would be spent on preprocessing the data or reproducing the algorithm in the paper.

### The solution

The Brainda will:

* Allow users to load the data easily without knowing the details
* Provide flexible hook functions to control the preprocessing flow
* Provide the latest decoding algorithms

The goal of the Brainda is to make researchers focus on improving their own BCI algorithms without wasting too much time on preliminary preparations.

## Features

* Improvements to MOABB APIs
   - add hook functions to control the preprocessing flow more easily
   - use joblib to accelerate the data loading
   - add proxy options for network conneciton issues
   - add more information in the meta of data
   - other small changes

* Supported Datasets
   - MI Datasets
     - AlexMI
     - BNCI2014001, BNCI2014004
     - PhysionetMI, PhysionetME
     - Cho2017
     - MunichMI
     - Schirrmeister2017
     - Weibo2014
     - Zhou2016
   - SSVEP Datasets
     - Nakanishi2015
     - Wang2016
     - BETA

* Implemented BCI algorithms
   - Decomposition Methods
     - SPoC, CSP, MultiCSP and FBCSP
     - CCA, itCCA, MsCCA, ExtendCCA, ttCCA, MsetCCA, MsetCCA-R, TRCA, TRCA-R, SSCOR and TDCA
     - DSP
   - Manifold Learning
     - Basic Riemannian Geometry operations
     - Alignment methods
     - Riemann Procustes Analysis
   - Deep Learning
     - ShallowConvNet
     - EEGNet
     - ConvCA
     - GuneyNet
   - Transfer Learning
     - MEKT
     - LST

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/brainda.git
   ```
2. Change to the project directory
   ```sh
   cd brainda
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```

## Usage

### Data Loading

In basic case, we can load data with the recommended options from the dataset maker.
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
    n_jobs=None, 
    verbose=False)
print(X.shape)
print(meta)
```
If you don't have the dataset yet, the program would automatically download a local copy, generally in your `~/mne_data` folder. However, you can always download the dataset in advance and store it in your specific folder.
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
You can also choose channels, events, intervals, srate, and subjects yourself.
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
or use different intervals for events. In this case, X, y and meta should be returned in dict.
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
    n_jobs=None, 
    verbose=False)
print(X['right_hand'].shape, X['feet'].shape)
```

### Preprocessing

Here is the flow of `paradigm.get_data` function:

<p align="center">
    <img src="images/get_data_flow.jpg" height="150" style="width:100%;max-width:700px;">
</p>

brainda provides 3 hooks that enable you to control the preprocessing flow in `paradigm.get_data`. With these hooks, you can operate data just like MNE typical flow:

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
    n_jobs=None, 
    verbose=False)
```
If the dataset maker provides these hooks in the dataset, brainda would call these hooks implictly. But you can always replace them with the above code.

### Machine Learning Pipeline

Now it's time to do some real BCI algorithms. Here is a demo of CSP for 2-class MI:

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
    n_jobs=None, 
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
If everything is fine, you will get the accuracy about 0.75.

## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China

Dr.Lichao Xu is the main contributor to the Brainda repository.

## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation and technical writing and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Email: TBC_TJU_2022@163.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
