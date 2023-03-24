from metabci.brainda.algorithms.decomposition.csp import FBCSP
from scipy import signal
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.datasets.TffMiData import TffMiData
from metabci.brainda.paradigms.imagery import TffImagery
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from mne.filter import resample
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def on_raw_loaded(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


# 带通滤波


def bandpass(sig, freq0, freq1, sample_rate, axis=-1):
    wn1 = 2 * freq0 / sample_rate
    wn2 = 2 * freq1 / sample_rate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


# 训练模型


def train_model(X, y, sample_rate=1000):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=sample_rate)
    # 滤波
    # X = bandpass(X, 6, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # brainda.algorithms.decomposition.csp.MultiCSP
    wp = [(4, 8), (8, 12), (12, 30)]
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=256, order=4, rp=0.5)
    # model = make_pipeline(
    #     MultiCSP(n_components = 2),
    #     LinearDiscriminantAnalysis())
    model = make_pipeline(*[
        FBCSP(n_components=5,
              n_mutualinfo_components=4,
              filterbank=filterbank),
        SVC()
    ])
    # fit()训练模型
    model = model.fit(X, y)

    return model


# 预测标签


def model_predict(X, sample_rate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=sample_rate)
    # 滤波
    X = bandpass(X, 8, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels


# 计算离线正确率


def offline_validation(X, y, sample_rate=1000):
    y = np.reshape(y, (-1))

    kfold_accs = []
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)  # 留一法交叉验证
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=sample_rate)  # 训练模型
        p_labels = model_predict(X_test, srate=sample_rate, model=model)  # 预测标签
        kfold_accs.append(np.mean(p_labels == y_test))  # 记录正确率

    return np.mean(kfold_accs)


if __name__ == '__main__':
    sample_rate = 1000
    stimulate_interval = [(0, 4)]
    subjects = list(range(1, 2))
    events_list = ["left_hand", "right_hand"]

    dataset = TffMiData(
        subjects=subjects,
        sample_rate=sample_rate,
    )
    paradigm = TffImagery(
        channels=dataset.channels,
        events=events_list,
        intervals=stimulate_interval,
        srate=sample_rate
    )
    paradigm.register_raw_hook(on_raw_loaded)

    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=2,
        verbose=False)

    y = label_encoder(y, np.unique(y))
