import sys
import ui.untitled as tt

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    ui = tt.Ui_Form()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
