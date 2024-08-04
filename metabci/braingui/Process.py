import os

import mne
from PyQt5.QtWidgets import QWidget, QDialog, QFileDialog
from .Ui_Form.Processing.preprocess.Preprocess_Form import Ui_Form_preprocessing
from .Ui_Form.Processing.preprocess.data_cut import Ui_Dialog_datacut
from .Ui_Form.Processing.preprocess.downsample import Ui_Dialog_downsample
from .Ui_Form.Processing.preprocess.Filter import Ui_Dialog_filter
from .Ui_Form.Processing.data_analysis.data_analysis import Ui_Dialog_data_analysis
from .Function import Preprocess_function
# ------------------------------------预处理界面------------------------------------
class Preprocess_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.load_data = None
        self.folder_path = None
        self.channel_name = None
        self.fs = None
        self.Function = Preprocess_function()
        self.ui = Ui_Form_preprocessing()
        self.ui.setupUi(self)
        self.show()
        self.ui.textEdit_order.setPlainText('你好,欢迎使用！')

        # 功能
        self.ui.pushButton_datacut.clicked.connect(self.open_datacut)             # 连接打开数据裁剪窗口函数
        self.ui.pushButton_downsample.clicked.connect(self.open_downsample)       # 连接打开降采样处理窗口函数
        self.ui.pushButton_filter.clicked.connect(self.open_filter)               # 连接打开滤波处理窗口函数
        self.ui.pushButton_input_rawdata.clicked.connect(self.loaddata)        # 导入原始数据
        self.ui.pushButton_checkdata.clicked.connect(self.check_data)          # 查看数据
        self.ui.pushButton_visualisation.clicked.connect(self.view_data)       # 当前数据可视化
        self.ui.pushButton_opendatafile.clicked.connect(self.open_currnet_file)   # 打开当前数据文件夹

    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()
    # 函数：打开数据裁剪处理窗口
    def open_datacut(self):
        self.datacut_dialog = Datacut_widget()
        self.datacut_dialog.ui.pushButton_startcut.clicked.connect(lambda: self.cut_data(dialogclass=self.datacut_dialog.ui))
    # 函数：打开降采样处理窗口
    def open_downsample(self):
        self.downsample_dialog = Downsample_widget()
        self.downsample_dialog.ui.pushButton_start_downsample.clicked.connect(lambda: self.downsample_data(dialog_class=self.downsample_dialog.ui))
    # 函数：打开滤波处理窗口
    def open_filter(self):
        self.filter_dialog = Filter_widget()
        self.filter_dialog.ui.pushButton_start_filter.clicked.connect(lambda: self.filter_data(dialog_class=self.filter_dialog.ui))
    # 命令框添加文字功能
    def add_textedit(self, textedit, text):
        # 获取当前文本框的内容
        current_text = textedit.toPlainText()
        # 将新的路径追加到文本框中
        new_text = current_text + "\n" + text
        self.ui.textEdit_order.setPlainText(new_text)
    # 加载原始数据功能
    def loaddata(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)  # 只选择文件夹
        dialog.setOption(QFileDialog.ShowDirsOnly, True)  # 只显示文件夹
        if dialog.exec_():
            self.folder_path = dialog.selectedFiles()[0]  # 获取选择的文件夹路径
            self.load_data, self.fs, self.channel_name = self.Function.data_load(pathname=self.folder_path)
            self.add_textedit(textedit=self.ui.textEdit_order, text='导入数据：' + self.folder_path)
            print('导入数据：' + self.folder_path)
            print(type(self.load_data))
    # 查看数据内容功能
    def check_data(self):
        cheackdataset = self.Function.data_check(pathname=self.folder_path)
        data_shape = cheackdataset['data_shape']
        channel_names = cheackdataset['channel_names']
        label_list = cheackdataset['label_list']
        fs = cheackdataset['fs']
        channel_number = cheackdataset['channel_number']
        self.add_textedit(textedit=self.ui.textEdit_order, text='-------数据内容查看：')
        self.add_textedit(textedit=self.ui.textEdit_order, text='data_shape:'+data_shape)
        self.add_textedit(textedit=self.ui.textEdit_order, text='channel_names:'+channel_names)
        self.add_textedit(textedit=self.ui.textEdit_order, text='label_list:'+label_list)
        self.add_textedit(textedit=self.ui.textEdit_order, text='fs:'+fs)
        self.add_textedit(textedit=self.ui.textEdit_order, text='channel_number:'+channel_number)

    # 打开当前数据的文件夹功能
    def open_currnet_file(self):
        # 使用os.startfile在Windows中打开文件夹
        os.startfile(self.folder_path)

    # 滤波功能
    def filter_data(self, dialog_class):
        if dialog_class.radioButton_selfdefine_filter.isChecked():
            lowcut = int(dialog_class.lineEdit_lowfrequent.text())
            highcut = int(dialog_class.lineEdit_highfrequent.text())
            self.load_data = self.Function.butter_bandpass_filter(data=self.load_data, lowcut=lowcut, highcut=highcut,fs=self.fs, order=4)
            self.add_textedit(textedit=self.ui.textEdit_order, text='自定义滤波：' + '('+str(lowcut) + '--' + str(highcut)+')' + 'Hz')
            print('自定义滤波')
        elif dialog_class.radioButton_band_filter.isChecked():
            # self.load_data = self.Function.butter_bandpass_filter(data=self.load_data, )
            self.add_textedit(textedit=self.ui.textEdit_order, text='频段滤波：？？')
            print('频段滤波')

    # 降采样功能
    def downsample_data(self, dialog_class):
        if dialog_class.radioButton_mindownsample.isChecked():
            self.load_data, self.fs = self.Function.downsample_data(data=self.load_data, method='min', factor=int(dialog_class.lineEdit_factor.text()), fs=self.fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='min降采样：'+str(self.load_data.shape))
        elif dialog_class.radioButton_maxdownsample.isChecked():
            self.load_data, self.fs = self.Function.downsample_data(data=self.load_data, method='max', factor=int(dialog_class.lineEdit_factor.text()), fs=self.fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='max降采样：' + str(self.load_data.shape))
        elif dialog_class.radioButton_meandownsample.isChecked():
            self.load_data, self.fs = self.Function.downsample_data(data=self.load_data, method='mean', factor=int(dialog_class.lineEdit_factor.text()), fs=self.fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='mean降采样：' + str(self.load_data.shape))

    # 切割数据功能
    def cut_data(self, dialogclass):
        mark_list = []
        time_list = []
        if dialogclass.checkBox_1.isChecked():
            mark_list.append(1)
            time_list.append([float(dialogclass.lineEdit_1f.text()), float(dialogclass.lineEdit_1a.text())])
        if dialogclass.checkBox_2.isChecked():
            mark_list.append(2)
            time_list.append([float(dialogclass.lineEdit_2f.text()), float(dialogclass.lineEdit_2a.text())])
        if dialogclass.checkBox_3.isChecked():
            mark_list.append(3)
            time_list.append([float(dialogclass.lineEdit_3f.text()), float(dialogclass.lineEdit_3a.text())])
        if dialogclass.checkBox_4.isChecked():
            mark_list.append(4)
            time_list.append([float(dialogclass.lineEdit_4f.text()), float(dialogclass.lineEdit_4a.text())])
        if dialogclass.checkBox_5.isChecked():
            mark_list.append(5)
            time_list.append([float(dialogclass.lineEdit_5f.text()), float(dialogclass.lineEdit_5a.text())])
        if dialogclass.checkBox_6.isChecked():
            mark_list.append(6)
            time_list.append([float(dialogclass.lineEdit_6f.text()), float(dialogclass.lineEdit_6a.text())])
        if dialogclass.checkBox_s1.isChecked():
            mark_list.append(int(dialogclass.lineEdit_s1.text()))
            time_list.append([float(dialogclass.lineEdit_7f.text()), float(dialogclass.lineEdit_7a.text())])
        if dialogclass.checkBox_s2.isChecked():
            mark_list.append(int(dialogclass.lineEdit_s1.text()))
            time_list.append([float(dialogclass.lineEdit_8f.text()), float(dialogclass.lineEdit_8a.text())])

        self.Function.cut_data(pathname=self.folder_path, time=time_list, mark_list=mark_list, fs=self.fs, loaddata=self.load_data)

        self.add_textedit(textedit=self.ui.textEdit_order, text='数据分割：' + 'label' + str(mark_list))

    # 基线校正功能
    def baseline_correction_data(self):
        self.load_data = self.Function.baseline_correction(eeg_signals=self.load_data)
        self.add_textedit(textedit=self.ui.textEdit_order, text='均值基线校正')

    # 当前数据可视化
    def view_data(self):
        # 创建一个info字典，包含通道信息和采样频率
        info = mne.create_info(ch_names=self.channel_name, sfreq=1000, ch_types='eeg')
        # 使用NumPy数组创建MNE的Raw数据结构
        raw = mne.io.RawArray(data=self.load_data, info=info)
        raw.plot(block=True)  # block=True会阻塞执行，直到关闭绘图窗口
        self.add_textedit(textedit=self.ui.textEdit_order, text='数据可视化')




class Datacut_widget(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog_datacut()
        self.ui.setupUi(self)
        self.show()

class Downsample_widget(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog_downsample()
        self.ui.setupUi(self)
        self.show()

class Filter_widget(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog_filter()
        self.ui.setupUi(self)
        self.show()





# ---------------------------------------数据处理窗口----------------------------

class Data_Analysis_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog_data_analysis()
        self.ui.setupUi(self)
        self.show()



