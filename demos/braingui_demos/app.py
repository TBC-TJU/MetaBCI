import sys
from PyQt5.QtWidgets import QApplication
from metabci.braingui.Form_class import Navigator_Form

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Navigator_Form()
    sys.exit(app.exec_())








