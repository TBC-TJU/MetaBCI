<!-- PROJECT LOGO -->
[中文](README.zh-cn.md)
<br />
<p align="center">
  <!-- <a href="https://github.com/Mrswolf/brainda">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Brainda</h3>

  <p align="center">
    A Library of Datasets and Algorithms for Brain-Computer Interface
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
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#main-features">Main Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#data-loading">Data Loading</a></li>
            <li><a href="#preprocessing">Preprocessing</a></li>
            <li><a href="#machine-learning-pipeline">Machine Learning Pipeline</a></li>
        </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

When I was a noob in the Brain-Computer Interface, there are 3 things that annoyed me most:
1. inject the conductive jelly
2. preprocess the EEG data from different formats
3. copy and past the algorithm codes in MATLAB over and over again

For the first problem, I feel hopeless(maybe there is a chance to replace the stupid injection in 10 years?). For other questions, I may find answers in the Python Community.
When I started to learn Python and MNE, I began to build my framework to simplify the EEG data acquisition and preprocessing steps. Then I found [MOABB](https://github.com/NeuroTechX/moabb), which is obviously much more advanced than my simple framework, so I started to use MOABB to get the EEG data. I also found that Scikit-learn provides an elegant abstraction of implementing machine learning algorithms with 'fit and transform'. This allows me to reuse existing codes instead of copy-and-paste. 

Brainda is a combination of advantages of MOABB and other excellent packages. I created this package to collect EEG datasets and implement BCI algorithms for my research.

### Main Features
1. Improvements to MOABB APIs
   - add hook functions to control the preprocessing flow more easily
   - use joblib to accelerate the data loading
   - add proxy options for network conneciton issues
   - add more information in the meta of data
   - other small changes
2. Implemented BCI algorithms in Python
   - Decomposition Methods
     - CSP, MultiCSP and FBCSP for MI
     - ExtendCCA, TRCA, Ensemble TRCA, and SSCOR for SSVEP
   - Manifold Learning
     - Basic Riemannian Geometry operations
     - Alignment methods
     - Riemann Procustes Analysis
   - Deep Learning
     - EEGNet
   - Transfer Learning
     - MEKT


<!-- GETTING STARTED -->
## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Mrswolf/brainda.git
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

<!-- USAGE EXAMPLES -->
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
    n_jobs=-1, 
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
    n_jobs=-1, 
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
    n_jobs=-1, 
    verbose=False)
print(X['right_hand'].shape, X['feet'].shape)
```
### Preprocessing
Here is the flow of `paradigm.get_data` function:

<p align="center">
    <img src="images/get_data_flow.jpg" width="700" height="150">
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
    n_jobs=-1, 
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
If everything is ok, you will get the accuracy about 0.925.

<!-- _For more examples, please refer to the [Documentation](https://github.com/Mrswolf/brainda)_ -->

<!-- ROADMAP -->
## Roadmap
- add demos
- add documents
- more datasets for P300
- more BCI algorithms
  
See the [open issues](https://github.com/Mrswolf/brainda/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

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