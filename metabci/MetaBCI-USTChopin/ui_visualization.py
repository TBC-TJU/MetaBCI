import sys
import subprocess
import PyQt5.QtCore
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer, QThread, QTime, Qt, QPoint
import qtawesome as qta
from PyQt5.QtWidgets import QMessageBox,QTextBrowser,QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QStackedWidget, QLineEdit, QPushButton, QComboBox, QLabel
from PyQt5.QtGui import QPixmap, QTextCursor
from scipy import signal
import time
import random
import pygame
from pygame.locals import *

#from pylsl import StreamInfo, StreamOutlet
from neuracle_lib.dataServer import dataserver_thread
import numpy as np
from datetime import datetime
import serial
from neuracle_lib.triggerBox import TriggerBox,TriggerIn,PackageSensorPara
import threading
from joblib import dump,load
key_label = [5,3,5,3,5,3,1,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1,2,2,4,4,3,1,5,2,4,3,2,5,5,5,3,5,3,5,3,1,2,4,3,2,1,1]

class MyThread(QThread):
    signalForText = pyqtSignal(str)
    def __init__(self, model=None, parent=None):
        super(MyThread, self).__init__(parent)
        # 如果有参数，可以封装在类里面
        self.model = model
    
    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):     
        p = subprocess.Popen(['python','offline_test.py', '--model', self.model], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 通过成员变量传参
        
        while True:
            result = p.stdout.readline()
            
            #print("result{}".format(result))
            if result != b'':
                print(result.decode('gbk').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                self.write(result.decode('gbk').strip('\r\n'))
            else:
                break
        p.stdout.close()
        p.stderr.close()
        p.wait()
    
class MyThread_1(QThread):
    signalForText = pyqtSignal(str)
    def __init__(self, ip=None, port= None, parent=None):
        super(MyThread_1, self).__init__(parent)
        # 如果有参数，可以封装在类里面
        self.ip = ip
        self.port = port
    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):     
        print("run")
        p = subprocess.Popen(['python','ME_pygame_BrainStim.py', '--ip', self.ip, '--port',self.port], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 通过成员变量传参
        # try:

        #     # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     stdout, stderr = p.communicate()
        #     if p.returncode != 0:
        #         print(f"Error connecting to device: {stderr.decode()}")
        #     else:
        #         print(f"Successfully connected to device. Output: {stdout.decode()}")
        # except subprocess.SubprocessError as e:
        #     print(f"An error occurred: {e}")    
        
        while True:
            result = p.stdout.readline()
            
            #print("result{}".format(result))
            if result != b'':
                print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                self.write(result.decode('utf-8').strip('\r\n'))
            else:
                break
        p.stdout.close()
        p.stderr.close()
        p.wait()
        
from PyQt5.QtCore import QThread
import joblib  # 假设您使用joblib来加载训练好的模型

from PyQt5.QtCore import QThread
import joblib  # 假设您使用joblib来加载训练好的模型

class MyThread_2(QThread):
    signalForText = pyqtSignal(str)
    def __init__(self,  ip=None, port= None, model=None, parent=None):
        super(MyThread_2, self).__init__(parent)
        # 如果有参数，可以封装在类里面
        self.model = model
        self.ip = ip
        self.port = port
    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):     
        p = subprocess.Popen(['python','USTChopin_online.py', '--model', self.model,'--ip', self.ip, '--port', self.port], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 通过成员变量传参
        
        while True:
            result = p.stdout.readline()
            
            #print("result{}".format(result))
            if result != b'':
                print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                self.write(result.decode('utf-8').strip('\r\n'))
            else:
                break
        p.stdout.close()
        p.stderr.close()
        p.wait()
        
class OnlineTestingPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        self.index = 0
        self.thread_data_server =  None
        
        
    def init_ui(self):
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LDA", "SVM", "RandomForest", "MLP", "Unimodal", "Late_Fusion"])
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("请输入IP地址")
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("请输入端口号")
        # self.show_image_button = QPushButton("显示手势")
        # self.show_image_button.clicked.connect(self.show_next_image)
        
        self.start_test_button = QPushButton("开始测试")
        self.start_test_button.clicked.connect(self.start_testing)
        #self.accuracy_label = QLabel("准确率: -")
        self.runTextBrowser = QTextBrowser() # 运行结果放入文本浏览器中
        #self.runTextBrowser.setGeometry(PyQt5.QtCore.QRect(600, 400))
        self.runTextBrowser.setObjectName("runTextBrowser")
        self.exit_test_button = QPushButton("退出测试")
        self.exit_test_button.clicked.connect(self.exit_testing)
        self.images = [
            QPixmap("Stim_images/L1R1.jpg"),
            QPixmap("Stim_images/L2R2.jpg"),
            QPixmap("Stim_images/L3R3.jpg"),
            QPixmap("Stim_images/L4R4.jpg"),
            QPixmap("Stim_images/L5R5.jpg")
        ]

        self.current_image_index = key_label[0]-1
        # self.image_label.setPixmap(self.images[key_label[0]-1])
        # self.image_label_1.setPixmap(self.images[key_label[1]-1])
        layout = QVBoxLayout()
        layout.addWidget(self.model_combo)
        layout.addWidget(self.ip_input)
        layout.addWidget(self.port_input)
        
        
        layout.addWidget(self.start_test_button)
        layout.addWidget(self.runTextBrowser)
        layout.addWidget(self.exit_test_button)
        
        self.setLayout(layout)
        
        
    
        
    def start_testing(self):
        
        ip = self.ip_input.text()
        port = self.port_input.text()
        model = self.model_combo.currentText()
        print(f"尝试连接到 {ip}:{port}")
        try:
            self.t3 = MyThread_2(ip, port, model)
            self.t3.signalForText.connect(self.onUpdateText)
            self.t3.start()
        except Exception as e:
            raise e
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()
        
        
    def show_completion_message(self):
        # 显示完成歌曲的信息
        QMessageBox.information(self, "完成", "歌曲播放完成")
    def onUpdateText(self, text):     
        cursor = self.runTextBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.runTextBrowser.append(text)
        self.runTextBrowser.setTextCursor(cursor)
        self.runTextBrowser.ensureCursorVisible()
                
    

    def exit_testing(self):
        self.main_window.show_main_page()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("设备连接与测试界面")
        self.setGeometry(100, 100, 800, 600)  # 根据需要调整窗口大小

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_main_page()
        self.init_online_testing_page()

    def init_main_page(self):
        self.main_page = QWidget()
        self.main_layout = QVBoxLayout(self.main_page)

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("请输入IP地址")
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("请输入端口号")
        self.connect_button = QPushButton("连接设备并采集数据")
        self.connect_button.clicked.connect(self.on_connect_clicked)
         # 模型选择下拉框
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LDA", "SVM","RandomForest", "MLP", "Unimodal","Late_Fusion"])

        # 数据采集按钮,
        # 离线训练按钮
        self.train_button = QPushButton("离线训练")
        self.train_button.clicked.connect(self.train_model)
        self.go_to_online_testing_button = QPushButton("转到在线测试")
        self.go_to_online_testing_button.clicked.connect(self.go_to_online_testing)

        self.main_layout.addWidget(self.ip_input)
        self.main_layout.addWidget(self.port_input)
        self.main_layout.addWidget(self.connect_button)
        self.main_layout.addWidget(self.model_combo)
        #self.main_layout.addWidget(self.collect_data_button)
        self.main_layout.addWidget(self.train_button)
        self.main_layout.addWidget(self.go_to_online_testing_button)

        self.stacked_widget.addWidget(self.main_page)

    def init_online_testing_page(self):
        self.online_testing_page = OnlineTestingPage(self)
        self.stacked_widget.addWidget(self.online_testing_page)

    def on_connect_clicked(self):
        # 连接设备的逻辑
        ip = self.ip_input.text()
        port = self.port_input.text()
        print(f"尝试连接到 {ip}:{port}")
        try:
            self.t1 = MyThread_1(ip, port)
            self.t1.start()
        except Exception as e:
            raise e
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()
    # def collect_data(self):
    #     # 数据采集逻辑
    #     print("开始数据采集...")
    #     # 这里可以调用 subprocess 来执行外部脚本，例如：
        

    def train_model(self):
        # 离线训练逻辑
        selected_model = self.model_combo.currentText()
        print(f"开始使用 {selected_model} 进行离线训练...")
        # 这里添加离线训练的逻辑 
        try:
            self.t = MyThread(selected_model)
            # 线程信号绑定到负责写入文本浏览器的槽函数onUpdateText
        
            self.t.start()
        except Exception as e:
            raise e
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()
    def go_to_online_testing(self):
        self.stacked_widget.setCurrentIndex(1)

    def show_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())