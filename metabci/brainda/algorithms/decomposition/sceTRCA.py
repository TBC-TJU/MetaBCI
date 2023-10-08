from typing import Optional, List, Tuple, Any
import numpy as np
from numpy import ndarray
from scipy import linalg as sLA
from math import sqrt, pow
from abc import abstractmethod, ABCMeta


def sign_sta(
        x: float):
    """Standardization of decision coefficient based on sign(x).

    Args:
        x (float)

    Returns:
        y (float): y=sign(x)*x^2
    """
    x = np.real(x)
    return (abs(x) / x) * (x ** 2)


def combine_feature(
        features: List[ndarray],
        func: Any = sign_sta):
    """Coefficient-level integration.

    Args:
        features (List[float or int or ndarray]): Different features.
        func (function): Quantization function.

    Returns:
        coef (the same type with elements of features): Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


def combine_fb_feature(
        features: List[Any]):
    """Coefficient-level integration specially for filter-bank design.

    Args:
        features (List[Any]): Coefficient matrices of different sub-bands.

    Returns:
        coef (float): Integrated coefficients.

    """
    coef = np.zeros_like(features[0])
    for nf, feature in enumerate(features):
        coef += (pow(nf + 1, -1.25) + 0.25) * (feature ** 2)
    return coef


def pick_subspace(
        descend_order: List[Tuple[int, float]],
        e_val_sum: float,
        ratio: float):
    """Config the number of subspaces.

    Args:
        descend_order (List[Tuple[int,float]]): See it in solve_gep() or solve_ep().
        e_val_sum (float): Trace of covariance matrix.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.

    Returns:
        n_components (int): The number of subspaces.
    """
    temp_val_sum = 0.0
    for n_components, do in enumerate(descend_order):  # n_sp: n_subspace
        temp_val_sum += do[-1]
        if temp_val_sum > ratio * e_val_sum:
            return n_components + 1


def solve_gep(
        A: ndarray,
        B: ndarray,
        n_components: Optional[int] = None,
        ratio: float = 0.5,
        mode: Optional[str] = 'Max'):
    """Solve generalized problems | generalized Rayleigh quotient:
        f(w)=wAw^T/(wBw^T) -> Aw = lambda Bw -> B^{-1}Aw = lambda w

    Args:
        A (ndarray): (m,m).
        B (ndarray): (m,m).
        n_components (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    e_val_sum = np.sum(e_val)
    descend_order = sorted(enumerate(e_val), key=lambda x: x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if not n_components:
        n_components = pick_subspace(descend_order, e_val_sum, ratio)
    if mode == 'Min':
        return np.real(e_vec[:, w_index][:, n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:, w_index][:, :n_components].T)


def pearson_corr(
        X: ndarray,
        Y: ndarray):
    """Pearson correlation coefficient (1-D or 2-D).

    Args:
        X (ndarray): (..., n_points)
        Y (ndarray): (..., n_points). The dimension must be same with X.

    Returns:
        corrcoef (float)
    """
    # check if not zero_mean():
    # X,Y = zero_mean(X), zero_mean(Y)
    cov_xy = np.sum(X * Y)
    var_x = np.sum(X ** 2)
    var_y = np.sum(Y ** 2)
    corrcoef = cov_xy / sqrt(var_x * var_y)
    return corrcoef


# %% Basic TRCA object
class BasicTRCA(metaclass=ABCMeta):
    def __init__(self,
                 standard: Optional[bool] = True,
                 ensemble: Optional[bool] = True,
                 n_components: Optional[int] = 1,
                 ratio: float = 0.5):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass

    @abstractmethod
    def transform(self,
                  X_test: ndarray):
        """Calculating decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
        """
        pass

    @abstractmethod
    def predict(self,
                X_test: ndarray):
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        pass


class BasicFBTRCA(metaclass=ABCMeta):
    def __init__(self,
                 standard: Optional[bool] = True,
                 ensemble: Optional[bool] = True,
                 n_components: Optional[int] = 1,
                 n_bands: int = 1,
                 ratio: float = 0.5):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.n_bands = n_bands
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass

    def transform(self,
                  X_test: ndarray):
        """Using filter-bank algorithms to calculate decision coefficients.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
        """
        # apply model.predict() method in each sub-band
        self.fb_rou: List[Any] = [[] for nb in range(self.n_bands)]
        self.fb_erou: List[Any] = [[] for nb in range(self.n_bands)]
        self.sub_models: List[Any] = [[] for nb in range(self.n_bands)]

        for nb in range(self.n_bands):
            self.sub_models[nb] = SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb] = fb_results[0]
            self.fb_erou[nb] = fb_results[2]

        # integration of multi-bands' results
        self.rou = combine_fb_feature(self.fb_rou)
        self.erou = combine_fb_feature(self.fb_erou)

        return self.rou, self.erou

    def predict(self,
                X_test: ndarray):
        """Calculating the prediction labels based on the decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Nt*Nte,). Predict labels of sc-TRCA.
            y_ensemble (ndarray): (Nt*Nte,). Predict labels of sc-eTRCA.
        """
        # basic information
        n_test = X_test.shape[1]
        self.fb_y_standard: List[List] = [[] for nb in range(self.n_bands)]
        self.fb_y_ensemble: List[List] = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_y_standard[nb] = fb_results[1]
            self.fb_y_ensemble[nb] = fb_results[3]

        # integration of multi-bands' results
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        self.rou, self.erou = self.transform(X_test)
        for nte in range(n_test):
            self.y_standard[nte] = np.argmax(self.rou[nte, :])
            self.y_ensemble[nte] = np.argmax(self.erou[nte, :])
        return self.y_standard, self.y_ensemble


def sctrca_compute(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: Optional[int] = 1,
        ratio: float = 0.5):
    """(Ensemble) similarity-constrained TRCA (sc-(e)TRCA).

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: sc-(e)TRCA model (dict).
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average template.
        S (ndarray): (Ne,Nc,Nc). Covariance of template.
        u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for EEG signal.
        v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal signal.
        u_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for EEG signal.
        v_concat (ndarray): (Ne*Nk,2*Nh). Concatenated filter for sinusoidal signal.
        uX (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for EEG signal.
        vY (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for sinusoidal signal.
        euX (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for EEG signal.
        evY (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for sinusoidal signal.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_2harmonics = sine_template.shape[1]  # 2*Nh

    S = np.zeros((n_events, n_chans + n_2harmonics, n_chans + n_2harmonics))  # (Ne,Nc+2Nh,Nc+2Nh)
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        avg_template[ne] = np.mean(X_temp, axis=0)  # (Nc,Np)

        YY = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        XX = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for tt in range(train_trials):
            XX += X_temp[tt] @ X_temp[tt].T
        XmXm = avg_template[ne] @ avg_template[ne].T  # (Nc,Nc)
        XmY = avg_template[ne] @ sine_template[ne].T  # (Nc,2Nh)

        # block covariance matrix S: [[S11,S12],[S21,S22]]
        S[ne, :n_chans, :n_chans] = XmXm  # S11
        S[ne, :n_chans, n_chans:] = (1 - 1 / train_trials) * XmY  # S12
        S[ne, n_chans:, :n_chans] = S[ne, :n_chans, n_chans:].T  # S21
        S[ne, n_chans:, n_chans:] = YY  # S22

        # block covariance matrix Q: blkdiag(Q1,Q2)
        for ntr in range(n_train[ne]):
            Q[ne, :n_chans, :n_chans] += X_temp[ntr] @ X_temp[ntr].T  # Q1
        Q[ne, n_chans:, n_chans:] = train_trials * YY  # Q2

    # GEP | train spatial filters
    u, v, ndim, correct = [], [], [], [False for ne in range(n_events)]
    for ne in range(n_events):
        spatial_filter = solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        u.append(spatial_filter[:, :n_chans])  # (Nk,Nc)
        v.append(spatial_filter[:, n_chans:])  # (Nk,2Nh)
    u_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
    start_idx = 0
    for ne, dims in enumerate(ndim):
        u_concat[start_idx:start_idx + dims] = u[ne]
        v_concat[start_idx:start_idx + dims] = v[ne]
        start_idx += dims

    # signal templates
    uX, vY = [], []  # Ne*(Nk,Np)
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ avg_template[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ avg_template[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)

    # sc-(e)TRCA model
    model = {
        'Q': Q, 'S': S,
        'u': u, 'v': v, 'u_concat': u_concat, 'v_concat': v_concat,
        'uX': uX, 'vY': vY, 'euX': euX, 'evY': evY, 'correct': correct
    }
    return model


# %% similarity constrained (e)TRCA | sc-(e)TRCA
class SC_TRCA(BasicTRCA):
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et) for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble
        }

        # train sc-TRCA models & templates
        model = sctrca_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = model['Q'], model['S']
        self.u, self.v = model['u'], model['v']
        self.u_concat, self.v_concat = model['u_concat'], model['v_concat']
        self.uX, self.vY = model['uX'], model['vY']
        self.euX, self.evY = model['euX'], model['evY']
        self.correct = model['correct']
        return self

    def transform(self,
                  X_test: ndarray):
        """Using sc-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-TRCA.
                Not empty when self.standard is True.
            erou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-eTRCA.
                Not empty when self.ensemble is True.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching (2-step)
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou)
        self.rou_sin = np.zeros_like(self.rou)
        self.erou = np.zeros_like(self.rou)
        self.erou_eeg = np.zeros_like(self.rou)
        self.erou_sin = np.zeros_like(self.rou)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_standard = self.u[nem] @ X_test[nte]
                    self.rou_eeg[nte, nem] = pearson_corr(
                        X=temp_standard,
                        Y=self.uX[nem]
                    )
                    self.rou_sin[nte, nem] = pearson_corr(
                        X=temp_standard,
                        Y=self.vY[nem]
                    )
                    self.rou[nte, nem] = combine_feature([
                        self.rou_eeg[nte, nem],
                        self.rou_sin[nte, nem]
                    ])

        if self.ensemble:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_ensemble = self.u_concat @ X_test[nte]
                    self.erou_eeg[nte, nem] = pearson_corr(
                        X=temp_ensemble,
                        Y=self.euX[nem]
                    )
                    self.erou_sin[nte, nem] = pearson_corr(
                        X=temp_ensemble,
                        Y=self.evY[nem]
                    )
                    self.erou[nte, nem] = combine_feature([
                        self.erou_eeg[nte, nem],
                        self.erou_sin[nte, nem]
                    ])

        return self.rou, self.erou

    def predict(self,
                X_test: ndarray):
        """Calculating the prediction labels based on the decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Nt*Nte,). Predict labels of sc-TRCA.
            y_ensemble (ndarray): (Nt*Nte,). Predict labels of sc-eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        event_type = self.train_info['event_type']
        self.rou, self.erou = self.transform(X_test)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        if self.standard:
            for nte in range(n_test):
                self.y_standard[nte] = event_type[np.argmax(self.rou[nte, :])]
        if self.ensemble:
            for nte in range(n_test):
                self.y_ensemble[nte] = event_type[np.argmax(self.erou[nte, :])]

        return self.y_standard, self.y_ensemble


class FB_SC_TRCA(BasicFBTRCA):
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train filter-bank sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = X_train.shape[0]

        # train sc-TRCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template
            )
        return self
