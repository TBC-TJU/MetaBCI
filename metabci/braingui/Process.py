from PyQt5.QtWidgets import QWidget, QDialog
from .Ui_Form.Processing.preprocess.Preprocess_Form import Ui_Form_preprocessing
from .Ui_Form.Processing.preprocess.data_cut import Ui_Dialog_datacut
from .Ui_Form.Processing.preprocess.downsample import Ui_Dialog_downsample
from .Ui_Form.Processing.preprocess.Filter import Ui_Dialog_filter
from .Ui_Form.Processing.data_analysis.data_analysis import Ui_Dialog_data_analysis

# ------------------------------------预处理界面------------------------------------
class Preprocess_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form_preprocessing()
        self.ui.setupUi(self)
        self.show()

        # 功能
        self.ui.pushButton_datacut.clicked.connect(self.open_datacut)             # 连接打开数据裁剪窗口函数
        self.ui.pushButton_downsample.clicked.connect(self.open_downsample)       # 连接打开降采样处理窗口函数
        self.ui.pushButton_filter.clicked.connect(self.open_filter)               # 连接打开滤波处理窗口函数

    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()

    # 函数：打开数据裁剪处理窗口
    def open_datacut(self):
        self.datacut_dialog = Datacut_widget()

    # 函数：打开降采样处理窗口
    def open_downsample(self):
        self.downsample_dialog = Downsample_widget()

    # 函数：打开滤波处理窗口
    def open_filter(self):
        self.filter_dialog = Filter_widget()


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



