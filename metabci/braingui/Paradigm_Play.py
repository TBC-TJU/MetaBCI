import sys
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QWidget, QApplication
from .Ui_Form.Paradigm.Paradigm_MI import Ui_Form_MI
from .Function import send_command, Voice_thread
import os
current_dir_image = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "Images" + os.sep + "left_hand.png",
        )
current_dir_video = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "Videos" + os.sep + "left_hand.mp4",
        )
class Paradigm(QWidget):
    def __init__(self, model_path: str):
        super(Paradigm, self).__init__()
        self.ui = Ui_Form_MI()        # 实例化UI界面
        self.ui.setupUi(self)         # 添加界面的UI控件
        self.timer = QTimer(self)     # 初始化定时器
        self.init_Media()             # 初始化视频播放控件
        self.init_flag()              # 初始化范式标志位
        self.init_factor()            # 初始化范式的要素
        self.init_window()            # 初始化窗口
        self.init_model(model_path=model_path)
        print('init successfully')
    # 检测键盘按键函数
    def keyPressEvent(self, event):
        # 举例
        if event.key() == Qt.Key_Escape:
            print('ESC-Close Paradigm')
            self.end()
            self.close()
            # self.timer.singleShot(3000, self.close)
        elif event.key() == Qt.Key_A:
            print('Paradigm_start_A')
            self.start()
    # 初始化视频播放控件
    def init_Media(self):
        # 创建一个视频播放器实例
        self.player = QMediaPlayer()
        # 把视频播放器放入对应的组件中（PyQt5.QtMultimediaWidgets.QVideoWidget）
        self.player.setVideoOutput(self.ui.widget_display)
    # 初始化范式标志位
    def init_flag(self):
        self.Paradigm_play_flag: bool = False    # 初始化范式播放标志位
        self.Paradigm_phase_flag: str = 'begin'  # 初始化范式开始界面
        self.trial = 0  # 初始化实验次数标志位
    # 初始化范式的要素
    def init_factor(self):
        self.num: dict = {
            # 初始化范式各阶段的播放次数
            'begin': 1,
            'openeye': 1,
            'closeeye': 1,
            'wait': 1,
            'display_imagery_rest': 5,
            'end': 1
        }
        self.continue_time: dict ={
            # 初始化范式各阶段的持续时间
            'begin': 5000,
            'openeye': 10000,
            'closeeye': 10000,
            'wait': 5000,
            'display': 4000,
            'imagery': 6000,
            'rest': 4000,
            'end': 0
        }
        self.text: dict = {
            # 初始化范式各阶段的文字描述
            'begin': 'Default',
            'openeye': 'Default',
            'closeeye': 'Default',
            'wait': 'Default',
            'display': 'Default',
            'imagery': 'Default',
            'rest': 'Default',
            'end': 'Default'
        }
        self.voice: dict = {
            # 初始化范式各阶段的语音文字
            'begin': '欢迎使用',
            'openeye': '请睁眼',
            'closeeye': '请闭眼',
            'wait': '请睁眼，等待提示',
            'display': '根据提示进行抓握',
            'imagery': '请进行运动想象',
            'rest': '请休息',
            'end': '范式结束'
        }
        self.mark: dict = {
            # 初始化范式各阶段的语音文字
            'begin': [0x01, 0xE1, 0x01, 0x00, 0x11],
            'openeye': [0x01, 0xE1, 0x01, 0x00, 0x05],
            'closeeye': [0x01, 0xE1, 0x01, 0x00, 0x04],
            'wait': [0],
            'display': [0],
            'imagery': [0x01, 0xE1, 0x01, 0x00, 0x01],
            'rest': [0],
            'end': [0x01, 0xE1, 0x01, 0x00, 0x22],
            'Enable': False,
            'COM': 'COM3'
        }
        self.video: dict = {
            # 初始化范式各阶段的视频资源
            'begin': 'Default',
            'openeye': 'Default',
            'closeeye': 'Default',
            'wait': 'Default',
            'display': 'Default',
            'imagery': 'Default',
            'rest': 'Default',
            'end': 'Default'
        }
        self.picture: dict = {
            # 初始化范式各阶段的图片资源
            'begin': 'Default',
            'openeye': 'Default',
            'closeeye': 'Default',
            'wait': 'Default',
            'display': 'Default',
            'imagery': 'Default',
            'rest': 'Default',
            'end': 'Default'
        }
    # 初始化窗口的第一张
    def init_window(self):
        self.paradigm_play()
        # 初始化两个button
        self.ui.pushButton_close.clicked.connect(self.close)
        self.ui.pushButton_start.clicked.connect(self.start)
    # 初始化模式
    def init_model(self, model_path: str):
        print('模式为：' + model_path)

    # 范式开始播放函数
    def start(self):
        self.Paradigm_play_flag = True     # 将范式开始播放使能
        # 等待x秒后开始范式
        self.timer.singleShot(self.continue_time[self.Paradigm_phase_flag], self.phase_update)

    # 范式结束播放函数
    def end(self):
        self.Paradigm_play_flag = False

    def paradigm_play(self):
        # 开始界面
        if self.Paradigm_phase_flag == 'begin':
            self.ui.stackedWidget_MI.setCurrentIndex(0)
            if self.text['begin'] == 'Default':
                print('begin_text_Default')
            else:
                self.ui.label_begin.setText(self.text['begin'])
        # 睁眼界面
        elif self.Paradigm_phase_flag == 'openeye':
            self.ui.stackedWidget_MI.setCurrentIndex(1)
            if self.text['openeye'] == 'Default':
                print('openeye_text_Default')
            else:
                self.ui.label_openeye.setText(self.text['openeye'])
        # 闭眼界面
        elif self.Paradigm_phase_flag == 'closeeye':
            self.ui.stackedWidget_MI.setCurrentIndex(2)
            if self.text['closeeye'] == 'Default':
                print('closeeye_text_Default')
            else:
                self.ui.label_closeeye.setText(self.text['closeeye'])
        # 等待界面
        elif self.Paradigm_phase_flag == 'wait':
            self.ui.stackedWidget_MI.setCurrentIndex(3)
            if self.text['wait'] == 'Default':
                print('wait_text_Default')
            else:
                self.ui.label_wait.setText(self.text['wait'])
        # 演示界面
        elif self.Paradigm_phase_flag == 'display':
            self.ui.stackedWidget_MI.setCurrentIndex(4)
            if self.text['display'] == 'Default':
                print('display_text_Default')
            elif self.text['display'] != 'Default':
                self.ui.label_display.setText(self.text['display'])
            if self.video['display'] == 'Default':
                print('display_video_Default')
                self.player.setMedia(QMediaContent(QUrl(current_dir_video)))
                self.player.play()
            elif self.video['display'] != 'Default':
                self.player.setMedia(QMediaContent(QUrl(self.video['display'])))
                self.player.play()
        # 想象界面
        elif self.Paradigm_phase_flag == 'imagery':
            self.ui.stackedWidget_MI.setCurrentIndex(5)
            if self.text['imagery'] == 'Default':
                print('imagery_text_Default')
            elif self.text['imagery'] != 'Default':
                self.ui.label_imagery.setText(self.text['imagery'])
            if self.picture['imagery'] == 'Default':
                print('imagery_picture_Default')
                self.ui.label_imagery_picture.setPixmap(QPixmap(current_dir_image).scaled(self.ui.label_imagery_picture.size(), Qt.KeepAspectRatio))
            elif self.picture['imagery'] != 'Default':
                self.ui.label_imagery_picture.setPixmap(QPixmap(self.picture['imagery']).scaled(self.ui.label_imagery_picture.size(), Qt.KeepAspectRatio))
        # 休息界面
        elif self.Paradigm_phase_flag == 'rest':
            self.ui.stackedWidget_MI.setCurrentIndex(6)
            if self.text['rest'] == 'Default':
                print('rest_text_Default')
            else:
                self.ui.label_rest.setText(self.text['rest'])
        # 结束界面
        elif self.Paradigm_phase_flag == 'end':
            self.ui.stackedWidget_MI.setCurrentIndex(7)
            if self.text['end'] == 'Default':
                print('end_text_Default')
            elif self.text['end'] != 'Default':
                self.ui.label_end.setText(self.text['end'])
        # 语音功能 and 打标功能
        Voice_thread(self.voice[self.Paradigm_phase_flag]).start()
        send_command(self.mark[self.Paradigm_phase_flag], com=self.mark['COM'], enable=self.mark['Enable'])




    def phase_update(self):
        # 判断是否更新，更新为哪个阶段
        if self.Paradigm_play_flag and (self.trial < self.num['display_imagery_rest']):
            if self.Paradigm_phase_flag == 'begin':
                self.Paradigm_phase_flag = 'openeye'
            elif self.Paradigm_phase_flag == 'openeye':
                self.Paradigm_phase_flag = 'closeeye'
            elif self.Paradigm_phase_flag == 'closeeye':
                self.Paradigm_phase_flag = 'wait'
            elif self.Paradigm_phase_flag == 'wait':
                self.Paradigm_phase_flag = 'display'
            elif self.Paradigm_phase_flag == 'display':
                self.Paradigm_phase_flag = 'imagery'
            elif self.Paradigm_phase_flag == 'imagery':
                self.Paradigm_phase_flag = 'rest'
                self.trial = self.trial + 1
            elif self.Paradigm_phase_flag == 'rest':
                self.Paradigm_phase_flag = 'display'
            # n秒后更新下一阶段
            self.timer.singleShot(self.continue_time[self.Paradigm_phase_flag], self.phase_update)
        else:
            self.Paradigm_phase_flag = 'end'
        # 执行界面更新，等待下次更新
        print('To-->', self.Paradigm_phase_flag, '窗口')
        self.paradigm_play()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Paradigm(model_path='model')
    win.showFullScreen()
    sys.exit(app.exec_())
