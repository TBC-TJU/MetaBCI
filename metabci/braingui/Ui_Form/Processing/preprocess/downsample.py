# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'downsample.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_downsample(object):
    def setupUi(self, Dialog_downsample):
        Dialog_downsample.setObjectName("Dialog_downsample")
        Dialog_downsample.resize(400, 313)
        self.frame = QtWidgets.QFrame(Dialog_downsample)
        self.frame.setGeometry(QtCore.QRect(20, 20, 353, 209))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.radioButton_downsample = QtWidgets.QRadioButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton_downsample.setFont(font)
        self.radioButton_downsample.setObjectName("radioButton_downsample")
        self.verticalLayout_2.addWidget(self.radioButton_downsample, 0, QtCore.Qt.AlignHCenter)
        self.radioButton_meandownsample = QtWidgets.QRadioButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton_meandownsample.setFont(font)
        self.radioButton_meandownsample.setObjectName("radioButton_meandownsample")
        self.verticalLayout_2.addWidget(self.radioButton_meandownsample, 0, QtCore.Qt.AlignHCenter)
        self.radioButton_maxdownsample = QtWidgets.QRadioButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton_maxdownsample.setFont(font)
        self.radioButton_maxdownsample.setObjectName("radioButton_maxdownsample")
        self.verticalLayout_2.addWidget(self.radioButton_maxdownsample, 0, QtCore.Qt.AlignHCenter)
        self.radioButton_mindownsample = QtWidgets.QRadioButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.radioButton_mindownsample.setFont(font)
        self.radioButton_mindownsample.setObjectName("radioButton_mindownsample")
        self.verticalLayout_2.addWidget(self.radioButton_mindownsample, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
        self.lineEdit_factor = QtWidgets.QLineEdit(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEdit_factor.setFont(font)
        self.lineEdit_factor.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_factor.setObjectName("lineEdit_factor")
        self.horizontalLayout.addWidget(self.lineEdit_factor)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.pushButton_start_downsample = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_start_downsample.setFont(font)
        self.pushButton_start_downsample.setObjectName("pushButton_start_downsample")
        self.verticalLayout.addWidget(self.pushButton_start_downsample)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.pushButton_help = QtWidgets.QPushButton(Dialog_downsample)
        self.pushButton_help.setGeometry(QtCore.QRect(320, 280, 75, 23))
        self.pushButton_help.setObjectName("pushButton_help")
        self.pushButton_close = QtWidgets.QPushButton(Dialog_downsample)
        self.pushButton_close.setGeometry(QtCore.QRect(10, 260, 101, 41))
        self.pushButton_close.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_close.setFont(font)
        self.pushButton_close.setObjectName("pushButton_close")

        self.retranslateUi(Dialog_downsample)
        self.pushButton_close.clicked.connect(Dialog_downsample.close)
        QtCore.QMetaObject.connectSlotsByName(Dialog_downsample)

    def retranslateUi(self, Dialog_downsample):
        _translate = QtCore.QCoreApplication.translate
        Dialog_downsample.setWindowTitle(_translate("Dialog_downsample", "Dialog"))
        self.radioButton_downsample.setText(_translate("Dialog_downsample", "普通降采样"))
        self.radioButton_meandownsample.setText(_translate("Dialog_downsample", "均值降采样"))
        self.radioButton_maxdownsample.setText(_translate("Dialog_downsample", "最大降采样"))
        self.radioButton_mindownsample.setText(_translate("Dialog_downsample", "最小降采样"))
        self.label.setText(_translate("Dialog_downsample", "降采样倍数："))
        self.lineEdit_factor.setText(_translate("Dialog_downsample", "4"))
        self.pushButton_start_downsample.setText(_translate("Dialog_downsample", "开始降采样"))
        self.pushButton_help.setText(_translate("Dialog_downsample", "帮助"))
        self.pushButton_close.setText(_translate("Dialog_downsample", "关闭"))
