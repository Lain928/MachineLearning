import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ui.untitled as tt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

class win(QWidget,tt.Ui_Form):  #继承类
    def __init__(self):
        super().__init__()
        #self.resize(300,300)
        self.setupUi(self)   #执行类中的setupUi函数
        self.filepath = ''
        self.pushButton.clicked.connect(self.getfilepath)
        self.pushButton_2.clicked.connect(self.paint_3d)
        self.pushButton_3.clicked.connect(self.close)   # 点击按钮之后关闭窗口


    def getfilepath(self):
        # 获取文件夹
        #dir_path = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "C:\\")
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        self.lineEdit.setText(directory[0])


    def paint_3d(self):
        if self.filepath == '':
            return "请先选择处理文件！！！"

        datasets = pd.read_csv(self.filepath, sep=' ', header=None)
        datasets.columns = ['X', 'Y', 'Z']
        xdata = datasets.loc[:, 'X']
        ydata = datasets.loc[:, 'Y']
        zdata = datasets.loc[:, 'Z']
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata)
        ax.set_xlabel('经度')
        ax.set_ylabel('维度')
        ax.set_zlabel('高度')
        plt.show()


if __name__=='__main__':
    app=QApplication(sys.argv)
    w=win()
    w.show()
    sys.exit(app.exec_())
