# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Navigator_Form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Navigator_From(object):
    def setupUi(self, Navigator_From):
        Navigator_From.setObjectName("Navigator_From")
        Navigator_From.resize(850, 650)
        self.pushButton_monitors = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_monitors.setGeometry(QtCore.QRect(200, 150, 200, 150))
        self.pushButton_monitors.setMinimumSize(QtCore.QSize(93, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_monitors.setFont(font)
        self.pushButton_monitors.setIconSize(QtCore.QSize(20, 20))
        self.pushButton_monitors.setAutoRepeatInterval(100)
        self.pushButton_monitors.setObjectName("pushButton_monitors")
        self.pushButton_paradigms = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_paradigms.setGeometry(QtCore.QRect(430, 150, 200, 150))
        self.pushButton_paradigms.setMinimumSize(QtCore.QSize(93, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_paradigms.setFont(font)
        self.pushButton_paradigms.setIconSize(QtCore.QSize(20, 20))
        self.pushButton_paradigms.setAutoRepeatInterval(100)
        self.pushButton_paradigms.setObjectName("pushButton_paradigms")
        self.pushButton_control = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_control.setGeometry(QtCore.QRect(200, 330, 200, 150))
        self.pushButton_control.setMinimumSize(QtCore.QSize(93, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_control.setFont(font)
        self.pushButton_control.setIconSize(QtCore.QSize(20, 20))
        self.pushButton_control.setAutoRepeatInterval(100)
        self.pushButton_control.setObjectName("pushButton_control")
        self.pushButton_processing = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_processing.setGeometry(QtCore.QRect(430, 330, 200, 150))
        self.pushButton_processing.setMinimumSize(QtCore.QSize(93, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_processing.setFont(font)
        self.pushButton_processing.setIconSize(QtCore.QSize(20, 20))
        self.pushButton_processing.setAutoRepeatInterval(100)
        self.pushButton_processing.setObjectName("pushButton_processing")
        self.pushButton_back = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_back.setGeometry(QtCore.QRect(30, 580, 160, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_back.setFont(font)
        self.pushButton_back.setObjectName("pushButton_back")
        self.label_function = QtWidgets.QLabel(Navigator_From)
        self.label_function.setGeometry(QtCore.QRect(140, 50, 541, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_function.setFont(font)
        self.label_function.setAlignment(QtCore.Qt.AlignCenter)
        self.label_function.setObjectName("label_function")
        self.pushButton_quit = QtWidgets.QPushButton(Navigator_From)
        self.pushButton_quit.setGeometry(QtCore.QRect(660, 580, 160, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_quit.setFont(font)
        self.pushButton_quit.setObjectName("pushButton_quit")

        self.retranslateUi(Navigator_From)
        self.pushButton_quit.clicked.connect(Navigator_From.close)
        QtCore.QMetaObject.connectSlotsByName(Navigator_From)

    def retranslateUi(self, Navigator_From):
        _translate = QtCore.QCoreApplication.translate
        Navigator_From.setWindowTitle(_translate("Navigator_From", "Form"))
        self.pushButton_monitors.setText(_translate("Navigator_From", "监测功能"))
        self.pushButton_paradigms.setText(_translate("Navigator_From", "范式采集"))
        self.pushButton_control.setText(_translate("Navigator_From", "控制功能"))
        self.pushButton_processing.setText(_translate("Navigator_From", "离线处理"))
        self.pushButton_back.setText(_translate("Navigator_From", "返回"))
        self.label_function.setText(_translate("Navigator_From", "功能导航界面"))
        self.pushButton_quit.setText(_translate("Navigator_From", "退出"))

