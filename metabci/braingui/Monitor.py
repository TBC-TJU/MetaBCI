from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget
from .neuracle_lib.dataServer import dataserver_thread
from .Ui_Form.Monitor.BrainGraphMonitor import Ui_brainGraph_Form
from .Function import Process

class BrainGraphMonitor(QWidget):
    def __init__(self):
        super(BrainGraphMonitor, self).__init__()
        self.ui = Ui_brainGraph_Form()
        self.ui.setupUi(self)
        self.show()
        self.start_draw_flag = False  # 开始绘制标志位
        self.P = Process()
        self.init_mode()
        self.timer = QTimer(self)
        self.ui.textEdit_order.setPlainText('欢迎使用')     # 初始化命令框
        self.init_pushButton()
        self.init_comcobox()

        # 定义地形图每个导联的位置坐标，此处为全脑地形图的导联坐标
        self.ch_pos = [
            [-20, 76], [0, 80], [20, 76],
            [-38, 67], [-24, 62], [24, 62], [38, 67],
            [-57, 52], [-42, 49], [-27, 46], [-12, 44], [0, 43], [12, 44], [27, 46], [42, 49], [57, 52],
            [-70, 37], [-53, 33], [-32, 30], [-15, 28], [0, 27], [15, 28], [32, 30], [53, 33], [70, 37],
            [-80, 10], [-57, 10], [-36, 10], [-16, 10], [0, 10], [16, 10], [36, 10], [57, 10], [80, 10],
            [-75, -17], [-54, -13], [-35, -10], [-18, -8], [18, -8], [35, -10], [54, -13], [75, -17],
            [-66, -35], [-49, -32], [-25, -29], [0, -30], [25, -29], [49, -32], [66, -35],
            [-49, -56], [-33, -48], [-20, -51], [0, -53], [20, -51], [33, -48], [49, -56],
            [-28, -70], [0, -73], [28, -70]
        ]
        # 创建一个 MyFigure 实例
        self.canvas = self.ui.Figure_braingraph

    def init_mode(self):
        self.EEG_device_parameter: dict = {
            "mode": self.ui.comboBox_acquire_device.currentText(),  # 当前采集设备型号
            "t_buffer": 3,    # 缓存池大小
            "srate": 1000,     # 采样频率
            "n_chan": 59,         # 采集导联数
        }
        self.Draw_mode = self.ui.comboBox_braingraph.currentText()
        self.order_output(text="模式已更新")
    def init_pushButton(self):
        self.ui.pushButton_close.clicked.connect(self.close_draw)
        self.ui.pushButton_start.clicked.connect(self.start_draw)
        self.ui.pushButton_connect.clicked.connect(lambda: self.EEG_connect_device(mode=self.EEG_device_parameter))

    def init_comcobox(self):
        self.ui.comboBox_acquire_device.currentIndexChanged.connect(self.init_mode)
        self.ui.comboBox_braingraph.currentIndexChanged.connect(self.init_mode)

    def order_output(self, text):
        # 获取当前文本框的内容
        current_text = self.ui.textEdit_order.toPlainText()
        # 将新的路径追加到文本框中
        new_text = current_text + "\n" + text
        self.ui.textEdit_order.setPlainText(new_text)

    def EEG_connect_device(self, mode: dict):
        if mode["mode"] == "Neuracle_Start":
            # 创建采集设备的相关参数
            neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                            srate=mode["srate"], n_chan=mode["n_chan"], t_buffer=mode["t_buffer"])
            # 创建数据接收客户端
            self.thread_data_server = dataserver_thread(threadName='data_server', device=neuracle['device_name'],
                                                        n_chan=neuracle['n_chan'], hostname=neuracle['hostname'],
                                                        port=neuracle['port'], srate=neuracle['srate'],
                                                        t_buffer=neuracle["t_buffer"])
            self.thread_data_server.Daemon = True  # 设置数据接收客户端线程为守护线程
            notconnect = self.thread_data_server.connect()  # 检测采集设备是否连接正常
            if notconnect:
                self.order_output(text="无法连接到采集设备，请打开采集设备接口 ")
            else:
                self.thread_data_server.start()  # 开启数据接收客户端线程
                self.order_output(text=mode["mode"] + '脑电采集设备，已连接！')

        self.order_output(text=self.ui.comboBox_acquire_device.currentText() + "设备已连接")

    def EEG_data_receive(self, mode: dict):
        raw_data = None
        if mode["mode"] == "Neuracle_Start":
            nUpdate = self.thread_data_server.get_bufferNupdate()
            if nUpdate > (1 * mode["srate"] - 1):
                raw_data = self.thread_data_server.get_bufferData()
                # self.thread_data_server.set_bufferNupdate(0)        # 数据更新数标志位，清零
            return raw_data

    def update_topography(self):
        if self.start_draw_flag == True:
            # 实时获取原始数据
            raw_data = self.EEG_data_receive(mode=self.EEG_device_parameter)
            data, ch_pos = self.P.data_draw(data=raw_data, mode=self.Draw_mode, fs=1000, nperseg=512)
            # 绘制新的地形图
            self.canvas.mne_EEG_plot(data=data[0, :], ch_pos=ch_pos, range=1)
            self.order_output(text="已更新")
            # 创建一个 QTimer 实例，每两秒触发一次
            self.timer.singleShot(500, self.update_topography)
        elif self.start_draw_flag == False:
            self.order_output(text="停止绘制")

    def start_draw(self):
        self.start_draw_flag = True
        self.order_output(text="开始绘制")
        self.update_topography()

    def close_draw(self):
        self.start_draw_flag = False
        self.thread_data_server.stop()
        self.close()



