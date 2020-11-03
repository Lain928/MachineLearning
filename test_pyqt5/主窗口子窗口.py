import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from ui.main import Ui_Form
from ui.child import Ui_Child


class MainWindow(QMainWindow, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.m_btnMain.clicked.connect(self.onClicked)
        # 一定要在主窗口类的初始化函数中对子窗口进行实例化，如果在其他函数中实例化子窗口
        # 可能会出现子窗口闪退的问题
        self.ChildDialog = ChildWin()

    def onClicked(self):
        # print('打开子窗口！')
        self.ChildDialog.show()
        # 连接信号
        self.ChildDialog._signal.connect(self.getData)

    def getData(self, parameter):
        self.m_showText.setText(parameter)


class ChildWin(QMainWindow, Ui_Child):
    # 定义信号
    _signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(ChildWin, self).__init__()
        self.setupUi(self)

        self.m_btnChild.clicked.connect(self.slot1)

    def slot1(self):
        data_str = self.m_childlineEdit.text()
        # 发送信号
        self._signal.emit(data_str)

        self.m_childlineEdit.setText("")
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())