import torch
import torch.nn as nn
from typing import Optional
from functools import partial
import numpy as np
from scipy.stats import pearsonr
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from joblib import Parallel, delayed
from metabci.brainda.algorithms.decomposition import TRCA
from  metabci.brainda.algorithms.decomposition.cca import _trca_kernel

"""
task-related correlation analysis network (TRCA-Net). A neural network-based SSVEP decoding method termed task-related correlation analysis network (TRCA-Net) [1]_
which combines the operation of TRCA spatial filtering and deep neural network.

..[1] Yang D. et al. TRCA-Net: using TRCA filters to boost the SSVEP classification with convolutional neural network. 
Journal of neural engineering vol. 20,4 10.1088/1741-2552/ace380. 12 Jul. 2023, doi:10.1088/1741-2552/ace380
    
"""

class DNN(nn.Module):
    ""
    """ the DNN structure in the backend of the TRCA-Net 
    
    author : Dian Li <lidian_123@tju.edu.cn>

    Created on : 2025-02-15

    updata log :
        2025-03-20 by Dian Li <lidian_123@tju.edu.cn>
    
    Parameters
    ----------
    num_bands : int
        The number of filters banks.
    num_channels : int
        The number of channels.
    num_classes : int
        The number of classes.
    nsamples : int
        The number of sampling point.
    
    Attributes
    ----------
    band_merger : nn.Conv2d
        the first convolutional layer of the DNN which combine the information from various bands.
    block_1 : nn.Sequential
        the second convolutional layer of the DNN.
    block_2 : nn.Sequential
        the third convolutional layer of the DNN.
    block_3 : nn.Sequential
        the fourth convolutional layer of the DNN.
    out : nn.Linear
        the last linear layer of the DNN.
    
    References
    ----------
    ..[1] Yang D. et al. TRCA-Net: using TRCA filters to boost the SSVEP classification with convolutional neural network. 
    Journal of neural engineering vol. 20,4 10.1088/1741-2552/ace380. 12 Jul. 2023, doi:10.1088/1741-2552/ace380
    
    """
    def __init__(self, num_bands=1, num_channels=9, num_classes=10, nsamples=150):
        super(DNN, self).__init__()
        self.nsamples = nsamples
        self.band_merger = nn.Conv2d(
            in_channels=num_bands,
            out_channels=1,
            kernel_size=(1, 1),
            bias=False,
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=120,
                kernel_size=(num_channels, 1),
            ),
            nn.Dropout(p=0.6),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=120,
                out_channels=120,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.Dropout(p=0.6),
            nn.ReLU(),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=120,
                out_channels=120,
                kernel_size=(1, 10),
                padding='same'
            ),
            nn.Dropout(p=0.9),
        )
        self.out = nn.Linear(120 * int(self.nsamples / 2), num_classes)

    def forward(self, x):
        ""
        """ Forward propagation process of the deep learning model

        Parameters
        ----------
        x : torch.Tensor
            The input data, shape(batch_size, n_channels, n_samples).

        Returns
        ----------
        x : torch.Tensor
            Tensor processed by the network, shape(batch_size, classnum).

        """
        x = self.band_merger(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

def _trca_feature(
        X: ndarray,
        templates: ndarray,
        Us: ndarray,
        n_components: int = 1,
        ensemble: bool = True,
):
    features = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            features.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        a = U.T @ X
        features.append(a)
    return features


class TRCA2(BaseEstimator, TransformerMixin, ClassifierMixin, TRCA):
    """"""
    """The core idea of Task-Related Component Analysis (TRCA) algorithm is to extract task-related components by
    improving the repeatability between trials, specifically, the algorithm is based on inter-trial covariance matrix
    maximization to achieve the extraction of task-related components, which belongs to the supervised learning method[1]_.


    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.

    References
    ----------
    .. [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller using
        task-related component analysis. IEEE Transactions on Biomedical Engineering, 2018, 104-112.

    """

    def __init__(
            self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """"""
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)

        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        self.Us_ = np.stack([_trca_kernel(X[y == label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        ''
        """Transform X into refined features

        Parameters
        ----------
        X: ndarray
            EEG data in one specific band, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        Features: ndarray
            The spatially filtered features, shape(n_trials, num_classes, n_samples)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble

        Features = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        Features = np.stack(Features)
        return Features


class GetLoader(torch.utils.data.Dataset):
    def __init__(self,self_data,self_label,N_datas):
        self.data = self_data
        self.label = self_label
        self.lenth = N_datas
    def __getitem__(self, index):
        data = self.data[index, :, :, :]
        label = self.label[:, index]
        return data,label
    def __len__(self):
        return self.data.shape[0]

from torch.utils.data import DataLoader

class TRCANet(BaseEstimator, TransformerMixin, ClassifierMixin):
    ""
    """
    task-related correlation analysis network (TRCA-Net). A neural network-based SSVEP decoding method termed task-related correlation analysis network (TRCA-Net) [1]_
    which combines the operation of TRCA spatial filtering and deep neural network.

    discriminant compacted network (Dis-ComNet) [1]_.

    author: Dian Li <lidian_123@tju.edu.cn>

    Created on: 2025-02-15

    updata log:
        2025-03-20 by Dian Li <lidian_123@tju.edu.cn>
    
    
    Parameters
    ----------
    nclass : int
        the number of classes.
    
    Attributes
    ----------
    trca_estimator : TRCA2
        the TRCA spatial estimator in the frontend
    device : torch.device
        the device on which the network is trained
    criterion : torch.nn.CrossEntropyLoss
        the loss function
    classes_ : ndarray
        predictive labels, obtained from labeled data by removing duplicate elements from it.
    dnn_model : torch.nn.Module
        the neural network 

    References
    ----------
    ..[1] Yang D. et al. TRCA-Net: using TRCA filters to boost the SSVEP classification with convolutional neural network. 
    Journal of neural engineering vol. 20,4 10.1088/1741-2552/ace380. 12 Jul. 2023, doi:10.1088/1741-2552/ace380

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using Dis-ComNet

        from metabci.brainda.algorithms.deep_learning import trca_net
        trcanet = TRCANet(nclass=nclass)
        trcanet.fit(X_train , y_train)
        acc = trcanet.score(X_test , y_test)

    """
    def __init__(self,  nclass: int = 7):
        self.trca_estimator = TRCA2(n_components=1, ensemble=True)
        self.nclass = nclass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

    def fit(self, X: ndarray, y: ndarray) -> 'TRCANet':
        """
        model training

        Parameters
        ----------
        X : ndarray
            EEG data, shape(n_trials, n_bands, n_channels, n_samples).
        y : ndarray
            Label, shape(n_trials,)

        Returns
        ----------
        self : TRCANet
            The trained model.

        """
        self.classes_ = np.unique(y)
        traindata = self.trca_estimator.fit_transform(X, y)
        # (ndatas, nsubbands, nchans, nsamples)
        source_label = np.zeros((self.nclass, X.shape[0]))
        for idata in range(y.shape[-1]):
            source_label[int(y[idata]), idata] = 1
        EEG_torch_data = GetLoader(traindata, source_label,  X.shape[0])
        datas = DataLoader(EEG_torch_data, batch_size=500, shuffle=True, drop_last=False, num_workers=0)
        model = DNN(nsamples=X.shape[-1]).to(self.device)
        # train the model
        self.dnn_model = self.fit_deep_model(datas=datas, model=model)
        return self

    def transform(self, X: ndarray):
        testdata = self.trca_estimator.transform(X)
        return testdata

    def predict(self, X: ndarray):
        ""
        """Predict the labels

        Parameters
        ----------
        X : ndarray
            EEG data, shape(n_trials, n_bands, n_channels, n_samples).

        Returns
        ----------
        labels : ndarray
            Predicting labels, shape(n_trials,).
        """
        testdata = self.transform(X)
        testdata = torch.from_numpy(testdata).to(self.device).to(torch.float32)
        test_outs = self.dnn_model(testdata)
        # 返回置信度矩阵
        labels = self.classes_[torch.argmax(test_outs, dim=-1)]
        return labels


    def fit_deep_model(self, datas, model):
        ""
        """ first training stage. 

        Parameters
        ----------
        datas : Tuple
            Data and corresponding labels in sorted by GetLoader.
        model : torch.nn.Module
            the neural network
        
        Returns
        ----------
        model : torch.nn.Module
            the neural network after training stage.
        """

        for epoch in range(500):
            # print("第{}个 epoch:".format(epoch + 1))
            for batch_idx, (data, label) in enumerate(datas):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
                data = data.to(torch.float32)
                label = label.to(torch.float32)
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = model(data)
                # outputs = nn.functional.softmax(outputs)
                loss = self.criterion(outputs, label)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return model

