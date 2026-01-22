import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from torch.utils.data import DataLoader

"""
discriminant compacted network (Dis-ComNet). A neural network-based SSVEP decoding method termed Discriminant Compacted Network (Dis-ComNet) [1]_
which combines advantages of both spatial filtering methods and deep learning methods.
This method enhances SSVEP features using Global template alignment (GTA) and Discriminant Spatial Pattern (DSP), and then enploys a Compacted Temporal-Spatio module (CTSM) to extract finer features.

..[1] Dian L., et al. Enhancing detection of SSVEPs using discriminant compacted network. 
      Journal of neural engineering vol. 22,1 10.1088/1741-2552/adb0f2. 14 Feb. 2025, doi:10.1088/1741-2552/adb0f2
"""

class Net(nn.Module):
    """aaa"""
    """
    The Compacted temporal-spatio module (CTSM) employed in the backend of discriminant compacted network (Dis-ComNet) [1]_
    
    author : Dian Li <lidian_123@tju.edu.cn>

    Created on : 2025-02-15

    updata log :
        2025-03-20 by Dian Li <lidian_123@tju.edu.cn>
    
    Parameters
    ----------
    datalen : int
        The length of the SSVEP data.
    Nb : int
        The number of filters banks.
    Nk : int
        The number of subspaces of DSP spatial filters.
    
    Attributes
    ----------
    drop_out : float
        The drop out probability.
    classnum : int
        The number of classes.
    block_1 : nn.Sequential
        The first convolutional block.
    block_2 : nn.Sequential
        The second convolutional block.
    block_3 : nn.Sequential
        The third convolutional block.
    out_1 : nn.Linear
        The final fully connected layer.

    References
    ----------
    ..[1] Dian L., et al. Enhancing detection of SSVEPs using discriminant compacted network. 
      Journal of neural engineering vol. 22,1 10.1088/1741-2552/adb0f2. 14 Feb. 2025, doi:10.1088/1741-2552/adb0f2
    """
    def __init__(self, datalen, classnum=10, Nb=1, Nk=3):
        super(Net,self).__init__()
        self.drop_out = 0.2
        self.classnum = classnum
        self.datalen = datalen

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=Nb,
                out_channels=15,
                kernel_size=(1, 25),
                bias=False,
                padding='same'
            ),
            nn.BatchNorm2d(15),
            nn.ELU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=15,
                out_channels=30,
                kernel_size=(Nk, 1),
                groups=15,
                bias=False
            ),
            nn.BatchNorm2d(30),
            nn.ELU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=30,
                out_channels=12,
                kernel_size=(1, 25),
                padding='same',
            ),
            nn.BatchNorm2d(12),
            nn.ELU(),
        )

        self.out1 = nn.Linear(int(12 * int(self.datalen)),self.classnum)

    def forward(self,x):
        ""
        """ Forward propagation of the deep learning model
        
        Parameters
        ----------
        x : torch.Tensor
            The input data, shape(batch_size, n_channels, n_samples).
            
        Returns
        ----------
        x : torch.Tensor
            Tensor processed by the network, shape(batch_size, classnum).
            
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(-1,int(12 * int(self.datalen)))
        x = self.out1(x)
        return x

class DisComNet(BaseEstimator, TransformerMixin, ClassifierMixin):
    """aa"""
    """
    discriminant compacted network (Dis-ComNet) [1]_.

    author: Dian Li <lidian_123@tju.edu.cn>

    Created on: 2025-02-15

    updata log:
        2025-03-20 by Dian Li <lidian_123@tju.edu.cn>

    Parameters
    ----------
    datalen : int
        The length of the SSVEP data.
    Nb : int
        The number of filters banks.
    Nk : int
        The number of subspaces of DSP spatial filters.
    
    Attributes
    ----------
    device : torch.device
        The device on which the network is trained.
    W : ndarray
        The spatial filter shared by all classes calculated individually in each band, shape (n_bands, Nk, n_channels).
    Phi : ndarray
        The matrix obtained by global template alignment (GTA) calculated individually in each band, shape(n_channels, n_channels).
    classes_ : ndarray
        Label, shape(n_classes,)
    classes : int
        the number of classes.
    criterion : torch.nn.CrossEntropyLoss
        the cross entropy loss.
    ComNet : torch.nn.Module
        the deep learning network
        
    References
    ----------
    ..[1] Dian L., et al. Enhancing detection of SSVEPs using discriminant compacted network. 
        Journal of neural engineering vol. 22,1 10.1088/1741-2552/adb0f2. 14 Feb. 2025, doi:10.1088/1741-2552/adb0f2
        
    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using Dis-ComNet

        from metabci.brainda.algorithms.deep_learning import Dis_ComNet
        discomnet = DisComNet(datalen = X.shape[-1], Nk = Nk, Nb = Nb)
        discomnet.fit(X_train , y_train)
        acc = discomnet.score(X_test , y_test)
    """



    def __init__(self, datalen, Nb=1, Nk=3):
        self.datalen = datalen
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Nk = Nk
        self.Nb = Nb

    def fit(self, X, y=None) -> 'DisComNet':
        """aaa"""
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
        self : DisComNet
            The trained model.
        
        """

        self.W = np.zeros((X.shape[1], self.Nk, 8))
        self.Phi = np.zeros((X.shape[1], X.shape[2], X.shape[2]))
        traindata = np.zeros((X.shape[0], 1, self.Nk, 2 * X.shape[3]))
        for iBand in range(X.shape[1]):
            Xi = X[:, iBand, :, :]
            phi , align_Xi= self.alignment(Xi)
            X_a = np.concatenate((Xi, align_Xi), axis=-1)
            self.W[iBand, :, :] = self.get_filter(X_a, y).T
            self.Phi[iBand, :, :] = phi
            traindata[:, iBand, :, :] = self.get_filtered_data(X_a, self.W[iBand, :, :])

        self.classes_ = np.unique(y)
        nEvents = len(np.unique(y))
        self.classes = nEvents

        ran = np.random.randint(nEvents, size=(1,))[0]
        validation_X = traindata[ran * nEvents: (ran + 1 ) * nEvents, :, :, :]
        validation_y = y[ran * nEvents: (ran + 1 ) * nEvents]

        traindata = np.delete(traindata, np.arange(0, nEvents, 1), axis=0)
        y = np.delete(y, np.arange(0, nEvents, 1), axis=0)

        source_label = np.zeros((self.classes, traindata.shape[0]))
        for idata in range(y.shape[-1]):
            source_label[int(y[idata]), idata] = 1

        EEG_torch_data = GetLoader(traindata, source_label, X.shape[0])
        datas = DataLoader(EEG_torch_data, batch_size=500, shuffle=True, drop_last=False, num_workers=0)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.fit_deep_model(datas=datas, validation_X=validation_X, validation_y=validation_y)

        traindata = np.concatenate((traindata, validation_X), axis=0)
        y = np.concatenate((y, validation_y), axis=0)

        source_label = np.zeros((self.classes, traindata.shape[0]))
        for idata in range(y.shape[-1]):
            source_label[int(y[idata]), idata] = 1
        EEG_torch_data = GetLoader(traindata, source_label, X.shape[0])
        datas = DataLoader(EEG_torch_data, batch_size=500, shuffle=True, drop_last=False, num_workers=0)
        self.second_fit_deep_model(datas=datas)
        return self

    def predict(self, X):
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
        testdata = np.zeros((X.shape[0], X.shape[1], self.Nk, 2 * X.shape[3]))
        for iBand in range(X.shape[1]):
            for idata in range(X.shape[0]):
                testdata[idata, iBand, :, :] = self.W[iBand, :, :] @ np.concatenate((X[idata, iBand, :, :], self.Phi[iBand, :, :] @ X[idata, iBand, :, :]), axis=-1)
        testdata = torch.from_numpy(testdata).to(self.device).to(torch.float32)
        test_outs = self.ComNet(testdata)
        labels = self.classes_[torch.argmax(test_outs, dim=-1).cpu()]
        return labels

    def alignment(self, X):
        ""
        """ global template alignment (GTA) approach aims to align the original signals with the global mean templates to achieve maximum similarity. 
        
        Parameters
        ----------
        X : ndarray
            EEG data in one sub-band, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        phi : ndarray
            The matrix obtained by global template alignment (GTA) in one specific filter band, shape(n_channels, n_channels).
        alignment_data : ndarray
            The aligned data obtained by global template alignment (GTA) in one specific filter band, shape (n_trials, n_channels, n_samples).
        """
        ntrials = X.shape[0]
        alignment_data = np.zeros_like(X)
        all_class_center = np.mean(X, axis=0)
        A = all_class_center @ all_class_center.T
        B = np.zeros((X.shape[1], X.shape[1]))
        for jtrial in range(ntrials):
            B += X[jtrial, :, :] @ X[jtrial, :, :].T
        phi = A @ np.linalg.inv(B)
        for itrial in range(ntrials):
            alignment_data[itrial] = phi @ X[itrial, :, :]
        return phi, alignment_data

    def get_filter(self, X, y):
        ""
        """ get the DSP spatial filter. 

        Parameters
        ----------
        X : ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y : ndarray
            Label, shape(n_trials,)

        Returns
        ----------
        wn : ndarray
            DSP spatial filter in one filter band , shape(Nk, n_channels).
        
        """
        label = np.unique(y)
        nTrials, nChans, nSamples = X.shape
        templates = []
        for i, ilabel in enumerate(label):
            label_idx = np.where(y == ilabel)[0]
            templates.append(np.mean(X[label_idx, :, :], 0))
        templates = np.array(templates)
        all_class_center = np.mean(templates, axis=0)
        Sb = np.zeros((nChans, nChans))
        Sw = np.zeros((nChans, nChans))
        for i, ilabel in enumerate(label):
            Sb += (templates[i, :, :] - all_class_center) @ (templates[i, :, :] - all_class_center).T
            label_idx = np.where(y == ilabel)[0]
            for jtrial in range(len(label_idx)):
                Sw += (X[label_idx[jtrial], :, :] - templates[i, :, :]) @ (X[label_idx[jtrial], :, :] - templates[i, :, :]).T
        Sb /= len(label)
        Sw /= nTrials
        # object        Sb / Sw
        Sw_inv = np.linalg.inv(Sw)
        Sw_inv_Sb = Sw_inv @ Sb
        value, vector = np.linalg.eig(Sw_inv_Sb)
        vector = vector[:, value.argsort()[::-1]]
        wn = vector[:, :self.Nk]
        return wn

    def get_filtered_data(self, X, wn):
        ""
        """ get the DSP-spatially-filtered data in one filter band . 

        Parameters
        ----------
        X : ndarray
            EEG data in a specific band, shape(n_trials, n_channels, n_samples).
        wn : ndarray
            DSP spatial filter in a specific band, shape(Nk, n_channels).

        Returns
        ----------
        X : ndarray
            DSP spatially filtered data in one filter band , shape(Nk, n_channels).

        """
        nTrials, nChans, nSamples = X.shape
        filtered_data = np.zeros((nTrials, self.Nk, nSamples))
        for itrial in range(nTrials):
            filtered_data[itrial] = (wn) @ (X[itrial, :, :])
        return filtered_data

    def fit_deep_model(self, datas, validation_X, validation_y):
        ""
        """ first training stage. 

        Parameters
        ----------
        datas : Tuple
            Data and corresponding labels in sorted by GetLoader.
        validation_X : ndarray
            Data in validation set, shape(num_validation, num_bands, Nk, 2 * num_samples).
        validation_y : ndarray
            Label in validation set, shape(num_validation,).

        """
        self.train_loss = []
        self.validation_loss = []
        val_y_onehot = np.zeros((len(validation_y), self.classes))
        for idata in range(len(validation_y)):
            val_y_onehot[idata, int(validation_y[idata])] = 1
        validation_X = torch.from_numpy(validation_X).to(self.device).to(torch.float32)
        val_y_onehot = torch.from_numpy(val_y_onehot).to(self.device).to(torch.float32)
        model = Net(2 * self.datalen).to(self.device)
        mini_loss = 10
        for epoch in range(500):
            for batch_idx, (data, label) in enumerate(datas):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
                data = data.to(torch.float32)
                label = label.to(torch.float32)
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = model(data)
                loss = self.criterion(outputs, label)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                val_loss = self.criterion(model(validation_X), val_y_onehot)
                self.train_loss.append(loss.item())
                self.validation_loss.append(val_loss.item())
                if val_loss.item() < mini_loss:
                    mini_loss = val_loss.item()
                    self.ComNet = model

    def second_fit_deep_model(self, datas):
        ""
        """ second training stage. 
        
        Parameters
        ----------
        datas : Tuple
            Data and corresponding labels in sorted by GetLoader.
        """
        for epoch in range(200):
            for batch_idx, (data, label) in enumerate(datas):
                optimizer = torch.optim.Adam(self.ComNet.parameters(), lr=0.0001, weight_decay=0.001)
                data = data.to(torch.float32)
                label = label.to(torch.float32)
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = self.ComNet(data)
                loss = self.criterion(outputs, label)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, self_data, self_label, N_datas):
        self.data = self_data
        self.label = self_label
        self.lenth = N_datas

    def __getitem__(self, index):
        data = self.data[index, :, :, :]
        label = self.label[:, index]
        return data, label

    def __len__(self):
        return self.data.shape[0]
