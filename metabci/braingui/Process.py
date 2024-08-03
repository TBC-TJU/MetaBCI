from PyQt5.QtCore import QDir
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
        self.ui.pushButton_checkdata.clicked.connect(self.check_data)

    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()
    # 函数：打开数据裁剪处理窗口
    def open_datacut(self):
        self.datacut_dialog = Datacut_widget()
    # 函数：打开降采样处理窗口
    def open_downsample(self):
        self.downsample_dialog = Downsample_widget()
        self.downsample_dialog.ui.pushButton_start_downsample.clicked.connect(lambda: self.downsample_data(dialog_class=self.downsample_dialog.ui))
    # 函数：打开滤波处理窗口
    def open_filter(self):
        self.filter_dialog = Filter_widget()
    # 文本框添加文字
    def add_textedit(self, textedit, text):
        # 获取当前文本框的内容
        current_text = textedit.toPlainText()
        # 将新的路径追加到文本框中
        new_text = current_text + "\n" + text
        self.ui.textEdit_order.setPlainText(new_text)
    # 加载原始数据
    def loaddata(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)  # 只选择文件夹
        dialog.setOption(QFileDialog.ShowDirsOnly, True)  # 只显示文件夹
        if dialog.exec_():
            self.folder_path = dialog.selectedFiles()[0]  # 获取选择的文件夹路径
            self.load_data = self.Function.data_load(pathname=self.folder_path)
            self.add_textedit(textedit=self.ui.textEdit_order, text='导入数据：' + self.folder_path)
            print('导入数据：' + self.folder_path)
            print(type(self.load_data))
    # 查看数据内容
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

    # 滤波功能
    def filter_data(self):
        ...
    # 降采样功能
    def downsample_data(self, dialog_class ,fs=1000):
        if dialog_class.radioButton_mindownsample.isChecked():
            self.load_data, _ = self.Function.downsample_data(data=self.load_data, method='min', factor=int(dialog_class.lineEdit_factor.text()), fs=fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='min降采样-->'+str(self.load_data.shape))
        elif dialog_class.radioButton_maxdownsample.isChecked():
            self.load_data, _ = self.Function.downsample_data(data=self.load_data, method='max', factor=int(dialog_class.lineEdit_factor.text()), fs=fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='max降采样-->' + str(self.load_data.shape))
        elif dialog_class.radioButton_meandownsample.isChecked():
            self.load_data, _ = self.Function.downsample_data(data=self.load_data, method='mean', factor=int(dialog_class.lineEdit_factor.text()), fs=fs)
            self.add_textedit(textedit=self.ui.textEdit_order, text='mean降采样-->' + str(self.load_data.shape))





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



