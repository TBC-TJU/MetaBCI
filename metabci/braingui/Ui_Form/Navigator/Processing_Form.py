# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Processing_Form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Processing_Form(object):
    def setupUi(self, Processing_Form):
        Processing_Form.setObjectName("Processing_Form")
        Processing_Form.resize(430, 480)
        self.pushButton_preprocessing = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_preprocessing.setGeometry(QtCore.QRect(140, 40, 160, 80))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_preprocessing.setFont(font)
        self.pushButton_preprocessing.setObjectName("pushButton_preprocessing")
        self.pushButton_feature_extraction = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_feature_extraction.setGeometry(QtCore.QRect(140, 220, 160, 80))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_feature_extraction.setFont(font)
        self.pushButton_feature_extraction.setObjectName("pushButton_feature_extraction")
        self.pushButton_model_training = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_model_training.setGeometry(QtCore.QRect(140, 320, 160, 80))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_model_training.setFont(font)
        self.pushButton_model_training.setObjectName("pushButton_model_training")
        self.pushButton_processing_back_navigator = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_processing_back_navigator.setGeometry(QtCore.QRect(20, 420, 171, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_processing_back_navigator.setFont(font)
        self.pushButton_processing_back_navigator.setObjectName("pushButton_processing_back_navigator")
        self.pushButton_processing_help = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_processing_help.setGeometry(QtCore.QRect(320, 430, 93, 28))
        self.pushButton_processing_help.setObjectName("pushButton_processing_help")
        self.pushButton_data_analysis = QtWidgets.QPushButton(Processing_Form)
        self.pushButton_data_analysis.setGeometry(QtCore.QRect(140, 130, 160, 80))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_data_analysis.setFont(font)
        self.pushButton_data_analysis.setObjectName("pushButton_data_analysis")

        self.retranslateUi(Processing_Form)
        QtCore.QMetaObject.connectSlotsByName(Processing_Form)

    def retranslateUi(self, Processing_Form):
        _translate = QtCore.QCoreApplication.translate
        Processing_Form.setWindowTitle(_translate("Processing_Form", "Form"))
        self.pushButton_preprocessing.setText(_translate("Processing_Form", "预处理"))
        self.pushButton_feature_extraction.setText(_translate("Processing_Form", "特征提取"))
        self.pushButton_model_training.setText(_translate("Processing_Form", "模型训练"))
        self.pushButton_processing_back_navigator.setText(_translate("Processing_Form", "返回导航界面"))
        self.pushButton_processing_help.setText(_translate("Processing_Form", "帮助"))
        self.pushButton_data_analysis.setText(_translate("Processing_Form", "数据分析"))

