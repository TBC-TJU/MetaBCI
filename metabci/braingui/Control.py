import sys

import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication
from .Paradigm_Play import Paradigm
from .Ui_Form.Control.offline_control import Ui_offline_Form
from .Ui_Form.Control.online_control import Ui_online_Form
from .neuracle_lib.dataServer import dataserver_thread
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from metabci.brainda.algorithms.decomposition import FBCSP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank

class Offline_control_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_offline_Form()
        self.ui.setupUi(self)
        self.show()


    def init_Button_Function(self):
        ...
    def robot_connect(self):
        ...
    def car_connect(self):
        ...
    def robot_control(self):
        ...
    def car_control(self):
        ...





class Online_control_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_online_Form()
        self.ui.setupUi(self)
        self.ui.textEdit_order.setPlainText('欢迎使用')   # 初始化命令框
        self.init_mode()                   # 初始化设备参数
        self.init_Button_Function()        # 初始化button控件功能
        self.init_Comcobox_Function()      # 初始化下拉框的控件功能
        self.show()
        wp = [(4, 8), (8, 12), (12, 30)]
        ws = [(2, 10), (6, 14), (10, 32)]
        filterbank = generate_filterbank(wp, ws, srate=128, order=4, rp=0.5)

        dataset = AlexMI()
        paradigm = MotorImagery(
            channels=None,
            events=['right_hand', 'feet'],
            intervals=[(0, 3)],  # 3 seconds
            srate=128
        )
        paradigm.register_raw_hook(self.raw_hook)
        paradigm.register_epochs_hook(self.epochs_hook)
        paradigm.register_data_hook(self.data_hook)

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

        # FBCSP with SVC classifier
        estimator = make_pipeline(*[
            FBCSP(n_components=5, n_mutualinfo_components=4, filterbank=filterbank),
            SVC()
        ])

        accs = []
        for k in range(kfold):
            train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
            # merge train and validate set
            train_ind = np.concatenate((train_ind, validate_ind))
            self.model = estimator.fit(X[train_ind], y[train_ind])

        self.order_output('FBCSP分类模型加载完成')

    # add 6-30Hz bandpass filter in raw hook
    def raw_hook(self, raw, caches):
        # do something with raw object
        raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
                   phase='zero-double')
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches

    def epochs_hook(self, epochs, caches):
        # do something with epochs object
        print(epochs.event_id)
        caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
        return epochs, caches

    def data_hook(self, X, y, meta, caches):
        # retrive caches from the last stage
        print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
        # do something with X, y, and meta
        caches['data_stage'] = caches.get('data_stage', -1) + 1
        return X, y, meta, caches

    def init_mode(self):
        self.EEG_device_parameter: dict = {
            "mode": self.ui.comboBox_gettype.currentText(),  # 当前采集设备型号
            "t_buffer": 6,    # 缓存池大小
            "srate": 1000,     # 采样频率
            "n_chan": 64,         # 采集导联数
        }
        self.Robot_device_parameter:dict = {
            "mode": self.ui.comboBox_robot_model.currentText()
        }
        self.Paradigm_parameter:dict = {
            "mode": self.ui.comboBox_paradigm_model.currentText()
        }
        self.order_output(text="----------模式更新------------")
        self.order_output(text="采集设备模式已更新:" + self.EEG_device_parameter["mode"])
        self.order_output(text="控制设备模式已更新:" + self.Robot_device_parameter["mode"])
        self.order_output(text = "范式模式已更新:" + self.Paradigm_parameter["mode"])
    def init_Button_Function(self):
        self.ui.pushButton_connect_get.clicked.connect(lambda: self.EEG_device_connect(mode=self.EEG_device_parameter))
        self.ui.pushButton_connect_robot.clicked.connect(self.robot_connect)
        self.ui.pushButton_open_paradigm.clicked.connect(lambda: self.Paradigm_start(mode=self.Paradigm_parameter))
        self.order_output(text="按键控件功能初始化完成")
    def init_Comcobox_Function(self):
        self.ui.comboBox_gettype.currentIndexChanged.connect(self.init_mode)
        self.ui.comboBox_robot_model.currentIndexChanged.connect(self.init_mode)
        self.ui.comboBox_paradigm_model.currentIndexChanged.connect(self.init_mode)
        self.order_output(text="选择框控件功能初始化完成")
    def order_output(self, text):
        # 获取当前文本框的内容
        current_text = self.ui.textEdit_order.toPlainText()
        # 将新的路径追加到文本框中
        new_text = current_text + "\n" + text
        self.ui.textEdit_order.setPlainText(new_text)
    def EEG_device_connect(self, mode: dict):
        # 1、Neuracle_W_start设备连接方法
        if mode["mode"] == "Neuracle_W_start":
            # 创建采集设备的相关参数
            neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                            srate=mode["srate"], n_chan=mode["n_chan"], t_buffer=mode["t_buffer"])
            # 创建数据接收客户端
            self.thread_data_server = dataserver_thread(threadName='data_server', device=neuracle['device_name'],
                                                   n_chan=neuracle['n_chan'], hostname=neuracle['hostname'],
                                                        port=neuracle['port'], srate=neuracle['srate'], t_buffer=neuracle["t_buffer"])
            self.thread_data_server.Daemon = True        # 设置数据接收客户端线程为守护线程
            notconnect = self.thread_data_server.connect()           # 检测采集设备是否连接正常
            if notconnect:
                self.order_output(text="无法连接到采集设备，请打开采集设备接口 ")
            else:
                self.thread_data_server.start()        # 开启数据接收客户端线程
                self.order_output(text=mode["mode"] + '脑电采集设备，已连接！')
        # 2、JellyFish_W4设备连接方法
        elif mode["mode"] == "JellyFish_W4":
            self.order_output(mode["mode"] + '脑电采集设备，已连接！')
        # 3、TGAM脑电采集设备连接
        elif mode["mode"] == "TGAM":
            self.order_output(text=mode["mode"] + '脑电采集设备，已连接！')
    def robot_connect(self):
        self.order_output(text="控制设备已连接")
    def Paradigm_start(self, mode: dict):
        print('开始1')
        self.Paradigm = Paradigm(model_path='model')
        print('开始2')
        self.Paradigm.show()
        self.control_paradigm(Paradigm_class=self.Paradigm)
        self.order_output(text=mode["mode"] + "范式已开启")
    def EEG_data_receive(self, mode: dict):
        raw_data = None
        if mode["mode"] == "Neuracle_W_start":
            nUpdate = self.thread_data_server.get_bufferNupdate()
            if nUpdate > (1 * mode["srate"] - 1):
                raw_data = self.thread_data_server.get_bufferData()
                self.thread_data_server.set_bufferNupdate(0)
            return raw_data
        elif mode["mode"] == "JellyFish_W4":
            return raw_data
        elif mode["mode"] == "TGAM":
            return raw_data

    def model_process(self, data):
        ...
    def robot_control(self, order: str):
        if order == 'left':
            self.order_output(text="左侧动作执行")
        elif order == 'right':
            self.order_output(text="右侧动作执行")
        elif order == 'stand':
            self.order_output(text="站立动作执行")
    def start_classify(self):
        # 1.实时获取数据
        # raw_data = self.EEG_data_receive(mode=self.EEG_device_parameter)[:59, :]
        # self.order_output(text=str(raw_data.shape))
        raw_data = np.random.rand(1, 15, 384)               # 模拟实时数据
        p_labels = self.model.predict(raw_data)
        self.order_output(text=p_labels + ' 右手运动')
    def control_paradigm(self, Paradigm_class):
        if Paradigm_class.control_start_flag == True:
            if Paradigm_class.control_flag == True:
                self.start_classify()
                Paradigm_class.control_flag = False
            elif Paradigm_class.control_flag == False:
                print("未检测到控制指令")
            self.Paradigm.timer.singleShot(300, lambda: self.control_paradigm(Paradigm_class=Paradigm_class))
        elif Paradigm_class.control_start_flag == False:
            self.order_output(text="Paradigm_close")