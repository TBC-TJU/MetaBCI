from PyQt5.QtWidgets import QWidget
from .Ui_Form.Control.offline_control import Ui_offline_Form
from .Ui_Form.Control.online_control import Ui_online_Form



class Offline_control_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_offline_Form()
        self.ui.setupUi(self)
        self.show()



class Online_control_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_online_Form()
        self.ui.setupUi(self)
        self.show()

