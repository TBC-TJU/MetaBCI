import sys
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
from .Ui_Form.Login.Login_Form import Ui_Login_From
from .Ui_Form.Navigator.Navigator_Form import Ui_Navigator_From
from .Ui_Form.Navigator.Control_Form import Ui_Control_Form
from .Ui_Form.Navigator.Monitor_Form import Ui_Monitors_Form
from .Ui_Form.Navigator.Paradigm_Form import Ui_Paradigm_From
from .Ui_Form.Navigator.Processing_Form import Ui_Processing_Form
from .Process import Preprocess_Form, Data_Analysis_Form
from .Control import Offline_control_Form, Online_control_Form
from .Paradigm_Play import Paradigm
from .Monitor import BrainGraphMonitor
from .Function import Form_QIcon

# 登录界面
class Login_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Login_From()     # 实例化界面
        self.ui.setupUi(self)         # 实例化界面的界面内容
        Form_QIcon(self)              # 设置软件图标
        self.show()                   # 界面展示

        # 功能
        self.ui.pushButton_login.clicked.connect(self.login)

    def login(self):
        if self.ui.lineEdit_name.text() == 'QLU' and self.ui.lineEdit_password.text() == '504':
            self.Navigator_form = Navigator_Form()
            self.close()
        else:
            QMessageBox.warning(self, '警告', '账号或密码错误，请重试', QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)
# 导航界面
class Navigator_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Navigator_From()
        self.ui.setupUi(self)  # 实例化界面的界面内容
        Form_QIcon(self)       # 设置软件图标
        self.show()
        # 功能
        self.ui.pushButton_monitors.clicked.connect(lambda: self.to_form(Monitor_Form))
        self.ui.pushButton_paradigms.clicked.connect(lambda: self.to_form(Paradigm_Form))
        self.ui.pushButton_control.clicked.connect(lambda: self.to_form(Control_Form))
        self.ui.pushButton_processing.clicked.connect(lambda: self.to_form(Processing_Form))
        self.ui.pushButton_back.clicked.connect(lambda: self.to_form(Login_Form))
        self.ui.pushButton_quit.clicked.connect(self.close)

    # 跳转到新界面
    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()
# 实时监测界面
class Monitor_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Monitors_Form()
        self.ui.setupUi(self)    # 实例化界面的界面内容
        Form_QIcon(self)         # 设置软件图标
        self.show()              # 界面展示
        # 功能
        self.ui.pushButton_monitor_back_nagivator.clicked.connect(lambda: self.to_form(Navigator_Form))
        self.ui.pushButton_brainGraph_monitor.clicked.connect(self.to_BrainGraphMonitor)

    # 跳转到新界面
    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()

    # 跳转到脑地形图监测界面
    def to_BrainGraphMonitor(self):
        self.braingraph = BrainGraphMonitor()
# 范式采集界面
class Paradigm_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Paradigm_From()
        self.ui.setupUi(self)   # 实例化界面的界面内容
        Form_QIcon(self)        # 设置软件图标
        self.show()             # 界面展示

        # 功能
        self.ui.pushButton_back_navigator.clicked.connect(lambda: self.to_form(Navigator_Form))
        self.ui.pushButton_MI_start.clicked.connect(self.to_MI_Paradigm)

    # 跳转到新界面
    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()
    # 转到MI范式播放界面
    def to_MI_Paradigm(self):
        self.Paradigm = Paradigm(model_path='model')
        self.Paradigm.showFullScreen()
# 控制界面
class Control_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Control_Form()
        self.ui.setupUi(self)  # 实例化界面的界面内容
        Form_QIcon(self)      # 设置软件图标
        self.show()  # 界面展示
        # 功能
        self.ui.pushButton_control_back_nagivator.clicked.connect(lambda: self.to_form(Navigator_Form))
        self.ui.pushButton_offline_control.clicked.connect(self.to_offline_control)
        self.ui.pushButton_online_control.clicked.connect(self.to_online_control)

    # 跳转到新界面
    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()

    def to_offline_control(self):
        self.offline_control_class = Offline_control_Form()

    def to_online_control(self):
        self.online_control_class = Online_control_Form()
# 离线处理界面
class Processing_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Processing_Form()
        self.ui.setupUi(self)  # 实例化界面的界面内容
        Form_QIcon(self)      # 设置软件图标
        self.show()
        # 功能
        self.ui.pushButton_processing_back_navigator.clicked.connect(lambda: self.to_form(Navigator_Form))
        self.ui.pushButton_preprocessing.clicked.connect(self.to_preprocess)
        self.ui.pushButton_data_analysis.clicked.connect(self.to_data_analysis)
    # 跳转到新界面
    def to_form(self, formClass):
        self.formclass = formClass()
        self.close()
    def to_preprocess(self):
        self.preprocess_class = Preprocess_Form()

    def to_data_analysis(self):
        self.data_analysis_class = Data_Analysis_Form()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Navigator_Form()
    sys.exit(app.exec_())

