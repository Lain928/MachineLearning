import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ui.untitled as tt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
import math
from ui.main import Ui_Main
from ui.mis_tar import Ui_mis_tar
from ui.threedim import Ui_three_dim
from ui.twodim import Ui_Two_Dim

box_items = ['导弹经度', '导弹纬度', '导弹高度', '导弹距离', '导弹速度', '导弹过载', '目标高度', '弹目距离', '目标经度', '目标纬度', '目标速度']


class summer:
    def __init__(self):
        self.filepath = ''
        self.getvalue_x = '导弹经度'
        self.getvalue_y = '导弹纬度'
        self.getvalue_z = '导弹高度'
        pass


    def getfilepath(self, num):
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        if num == 1:
            self.lineEdit_three.setText(directory[0])
        if num == 2:
            self.lineEdit_two.setText(directory[0])
        if num == 3:
            self.lineEdit_mis.setText(directory[0])


    def print_value_x(self, i):
        self.getvalue_x = i
    def print_value_y(self, i):
        self.getvalue_y = i
    def print_value_z(self, i):
        self.getvalue_z = i


class MainWindow(QMainWindow, Ui_Main):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.radioBtnThree.setChecked(True) # 默认是绘制三维图
        self.radioBtnTwo.setChecked(False)
        self.radioBtnMis_Tar.setChecked(False)
        self.pushBtnRun.clicked.connect(self.onClicked)
        self.pushBtnClose.clicked.connect(self.close)
        # self.m_btnMain.clicked.connect(self.onClicked)
        # 一定要在主窗口类的初始化函数中对子窗口进行实例化，如果在其他函数中实例化子窗口
        # 可能会出现子窗口闪退的问题
        self.ChildDialog = ChildWin()
        self.TwoDialog = TwoWin()
        self.ThreeDialog = ChildWin()
        self.MisTarDialog = Mis_Tar_Win()

    def onClicked(self):
        if self.radioBtnThree.isChecked() == True: # 三维图
            self.ChildDialog.show()
            pass
        if self.radioBtnTwo.isChecked() == True: # 二维图
            self.TwoDialog.show()
            pass
        if self.radioBtnMis_Tar.isChecked() == True: # 弹目交互
            self.MisTarDialog.show()
            pass


class ChildWin(QMainWindow, Ui_three_dim,summer):
    def __init__(self):
        super(ChildWin, self).__init__()
        self.setupUi(self)
        self.filepath = ''
        self.btn_three_choose.clicked.connect(lambda: self.getfilepath(1))
        self.btn_three_run.clicked.connect(self.plotting_3d)
        self.btn_three_close.clicked.connect(self.close)

        # 下拉框
        self.getvalue_x = '导弹经度'
        self.getvalue_y = '导弹纬度'
        self.getvalue_z = '导弹高度'
        self.Box_three_x.addItems(box_items)
        self.Box_three_y.addItems(box_items)
        self.Box_three_z.addItems(box_items)
        # 下拉框默认选项
        self.Box_three_x.setCurrentIndex(0)  # 设置默认值
        self.Box_three_y.setCurrentIndex(1)  # 设置默认值
        self.Box_three_z.setCurrentIndex(2)  # 设置默认值
        # 信号
        self.Box_three_x.currentIndexChanged[str].connect(self.print_value_x) # 条目发生改变，发射信号，传递条目内容
        self.Box_three_y.currentIndexChanged[str].connect(self.print_value_y) # 条目发生改变，发射信号，传递条目内容
        self.Box_three_z.currentIndexChanged[str].connect(self.print_value_z) # 条目发生改变，发射信号，传递条目内容

    def plotting_3d(self):
        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        rows = datasets.shape[0]
        xdata = datasets[self.getvalue_x]
        ydata = datasets[self.getvalue_y]
        zdata = datasets[self.getvalue_z]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata,color='red')
        ax.set_xlabel(self.getvalue_x)
        ax.set_ylabel(self.getvalue_y)
        ax.set_zlabel(self.getvalue_z)
        ax.set_title("数据分析")
        plt.show()

class Mis_Tar_Win(QMainWindow, Ui_mis_tar,summer):
    def __init__(self):
        super(Mis_Tar_Win, self).__init__()
        self.setupUi(self)
        self.filepath = ''
        self.btn_mis_choose.clicked.connect(lambda: self.getfilepath(3))
        self.btn_mis_run.clicked.connect(self.plotting_mis_tar)
        self.btn_mis_close.clicked.connect(self.close)


    def helper_poltting(self,dataset):
        xdata1 = dataset.loc[:, '导弹经度']/ 3.141593 * 180
        ydata1 = dataset.loc[:, '导弹纬度']/ 3.141593 * 180
        zdata1 = dataset.loc[:, '导弹高度']

        xdata2 = dataset.loc[:, '目标经度']/ 3.141593 * 180
        ydata2 = dataset.loc[:, '目标纬度']/ 3.141593 * 180
        zdata2 = dataset.loc[:, '目标高度']

        rows = dataset.shape[0]
        # 交汇点 导弹坐标信息
        last_mis_x = xdata1[rows - 1]
        last_mis_y = ydata1[rows - 1]
        last_mis_z = zdata1[rows - 1]
        # 交汇点 目标坐标信息
        last_tar_x = xdata2[rows - 1]
        last_tar_y = ydata2[rows - 1]
        last_tar_z = zdata2[rows - 1]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata1, ydata1, zdata1, "r", label="SM6")
        ax.plot(xdata2, ydata2, zdata2, "g", label="Target")
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_zlabel('高度')
        ax.set_title("弹目交互数据分析")

        # 对交汇点设置文本标注信息
        axis_tar = "爆炸点经度:" + str(round(last_tar_x, 2)) + "\n" \
                   + "爆炸点维度:" + str(round(last_tar_y, 2)) \
                   + "\n" + "爆炸点高度:" + str(round(last_tar_z))
        ax.text(last_tar_x, last_tar_y, last_tar_z, axis_tar, color='blue')
        ax.scatter(last_tar_x, last_tar_y, last_tar_z, marker="v", c="blue")

        plt.legend()
        plt.show()


    def plotting_mis_tar(self):
        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        self.helper_poltting(datasets)
        pass

class TwoWin(QMainWindow, Ui_Two_Dim,summer):
    def __init__(self):
        super(TwoWin, self).__init__()
        self.setupUi(self)
        self.filepath = ''
        self.btn_two_choose.clicked.connect(lambda: self.getfilepath(2))
        self.btn_two_run.clicked.connect(self.poltting_2d)
        self.btn_two_close.clicked.connect(self.close)

        # 下拉框
        self.Box_two_x.addItems(box_items)
        self.Box_two_y.addItems(box_items)
        # 下拉框默认选项
        self.Box_two_x.setCurrentIndex(0)  # 设置默认值
        self.Box_two_y.setCurrentIndex(1)  # 设置默认值
        # 信号 x y z
        self.Box_two_x.currentIndexChanged[str].connect(self.print_value_x)  # 条目发生改变，发射信号，传递条目内容
        self.Box_two_y.currentIndexChanged[str].connect(self.print_value_y)  # 条目发生改变，发射信号，传递条目内容


    def poltting_2d(self):
        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        xdata = datasets[self.getvalue_x]
        ydata = datasets[self.getvalue_y]
        plt.plot(xdata, ydata,color='red')
        plt.xlabel(self.getvalue_x)
        plt.ylabel(self.getvalue_y)
        plt.grid()
        plt.show()
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())