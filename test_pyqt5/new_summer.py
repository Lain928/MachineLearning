import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import ui.untitled as tt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog
from PyQt5.QtCore import Qt
import math
from ui.main import Ui_Main
from ui.mis_tar import Ui_mis_tar
from ui.threedim import Ui_three_dim
from ui.twodim import Ui_Two_Dim

from ui.new_main import Ui_UI_New
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


box_items = ['导弹经度', '导弹纬度', '导弹高度', '导弹距离', '导弹速度', '导弹过载', '目标高度', '弹目距离', '目标经度', '目标纬度', '目标速度','仿真时间']


class new_summer(QMainWindow, Ui_UI_New):
    def __init__(self):
        super(new_summer, self).__init__()
        self.setupUi(self)

        self.filepath = ''
        self.getvalue_x = '导弹经度'
        self.getvalue_y = '导弹纬度'
        self.getvalue_z = '导弹高度'

        # 初始化一些信息
        self.radio_three.setChecked(True) # 默认是绘制三维图
        self.radio_two.setChecked(False)
        self.radio_mis.setChecked(False)
        self.btn_run.clicked.connect(self.onClicked)
        self.btn_close.clicked.connect(self.close)
        self.btn_choosefile.clicked.connect(self.getfilepath)


        # 下拉框默认设置
        self.comboBox_x.addItems(box_items)
        self.comboBox_y.addItems(box_items)
        self.comboBox_z.addItems(box_items)
        # 下拉框默认选项
        self.comboBox_x.setCurrentIndex(0)  # 设置默认值
        self.comboBox_y.setCurrentIndex(1)  # 设置默认值
        self.comboBox_z.setCurrentIndex(2)  # 设置默认值
        # 信号
        self.comboBox_x.currentIndexChanged[str].connect(self.print_value_x) # 条目发生改变，发射信号，传递条目内容
        self.comboBox_y.currentIndexChanged[str].connect(self.print_value_y) # 条目发生改变，发射信号，传递条目内容
        self.comboBox_z.currentIndexChanged[str].connect(self.print_value_z) # 条目发生改变，发射信号，传递条目内容

        # radion 信号与槽
        self.radio_three.toggled.connect(self.use_3d)
        self.radio_two.toggled.connect(self.use_2d)
        self.radio_mis.toggled.connect(self.use_mis)
        pass


    def onClicked(self):
        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        rows = datasets.shape[0]

        if self.radio_three.isChecked() == True: # 三维图
            self.plotting_3d(datasets,rows)
            pass
        if self.radio_two.isChecked() == True: # 二维图
            self.poltting_2d(datasets,rows)
            pass
        if self.radio_mis.isChecked() == True: # 弹目交互
            self.poltting_mis(datasets,rows)
            pass


    def use_3d(self):
        self.label_x.setVisible(True) #  设置控件是否隐藏
        self.label_y.setVisible(True)
        self.label_z.setVisible(True)
        self.comboBox_x.setVisible(True)
        self.comboBox_y.setVisible(True)
        self.comboBox_z.setVisible(True)
        pass
    def use_2d(self):
        self.label_x.setVisible(True) #  设置控件是否隐藏
        self.label_y.setVisible(True)
        self.label_z.setHidden(True)
        self.comboBox_x.setVisible(True)
        self.comboBox_y.setVisible(True)
        self.comboBox_z.setHidden(True)
        pass
    def use_mis(self):# 弹目交互时 下拉框和标签都隐藏
        self.label_x.setHidden(True) #  设置控件是否隐藏
        self.label_y.setHidden(True)
        self.label_z.setHidden(True)
        self.comboBox_x.setHidden(True)
        self.comboBox_y.setHidden(True)
        self.comboBox_z.setHidden(True)
        pass


    def getfilepath(self):
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        self.lineEdit_path.setText(directory[0])


    def print_value_x(self, i):
        self.getvalue_x = i
    def print_value_y(self, i):
        self.getvalue_y = i
    def print_value_z(self, i):
        self.getvalue_z = i


    # 三维绘图
    def plotting_3d(self,datasets,rows):
        if self.getvalue_x == '导弹经度' or self.getvalue_x == '导弹纬度' \
                or self.getvalue_x == '目标纬度' or self.getvalue_x == '目标经度':
            # arr_x = datasets[self.getvalue_x].values / 3.1415927 * 180
            # xdata1 = np.linspace(arr_x[200:].min(),arr_x[200:].max(),rows-200)
            xdata = datasets[self.getvalue_x] /3.1415927 *180
        else:
            xdata = datasets[self.getvalue_x]
        if self.getvalue_y == '导弹经度' or self.getvalue_y == '导弹纬度' \
                or self.getvalue_y == '目标纬度' or self.getvalue_y == '目标经度':
            # arr_y = datasets[self.getvalue_y].values / 3.1415927 * 180
            # ydata1 = np.linspace(arr_y[200:].min(),arr_y[200:].max(),rows-200)
            ydata = datasets[self.getvalue_y]/3.1415927 *180
        else:
            ydata = datasets[self.getvalue_y]
        if self.getvalue_z == '导弹经度' or self.getvalue_z == '导弹纬度' \
                or self.getvalue_z == '目标纬度' or self.getvalue_z == '目标经度':
            # arr_z = datasets[self.getvalue_z].values / 3.1415927 * 180
            # zdata = np.linspace(arr_z.min(),arr_z.max(),rows)
            zdata = datasets[self.getvalue_z]/3.1415927 *180
        else:
            zdata = datasets[self.getvalue_z]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata,color='red')
        ax.set_xlabel(self.getvalue_x)
        ax.set_ylabel(self.getvalue_y)
        ax.set_zlabel(self.getvalue_z)
        ax.set_title("数据分析")
        plt.show()

    #弹目交互
    def poltting_mis(self,dataset,rows):
        xdata1 = dataset.loc[:, '导弹经度']/ 3.141593 * 180
        ydata1 = dataset.loc[:, '导弹纬度']/ 3.141593 * 180
        zdata1 = dataset.loc[:, '导弹高度']

        xdata2 = dataset.loc[:, '目标经度']/ 3.141593 * 180
        ydata2 = dataset.loc[:, '目标纬度']/ 3.141593 * 180
        zdata2 = dataset.loc[:, '目标高度']

        last_pos = dataset.loc[:, '弹目距离']

        # 交汇点 导弹坐标信息
        last_mis_x = xdata1[rows - 1]
        last_mis_y = ydata1[rows - 1]
        last_mis_z = zdata1[rows - 1]
        # 交汇点 目标坐标信息
        last_tar_x = xdata2[rows - 1]
        last_tar_y = ydata2[rows - 1]
        last_tar_z = zdata2[rows - 1]
        # 提取爆炸点的弹目距离，用于判断是否会爆炸
        last_pos_bao = last_pos[rows - 1]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata1, ydata1, zdata1, "r", label="SM6")
        ax.plot(xdata2, ydata2, zdata2, "g", label="Target")
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_zlabel('高度')
        ax.set_title("弹目交互数据分析")

        if last_pos_bao < 10:
            # 对交汇点设置文本标注信息 round(),函数为保留几位小数
            axis_tar = "爆炸点经度:" + str(round(last_tar_x, 2)) + "\n" + \
                       "爆炸点纬度:" + str(round(last_tar_y, 2)) + "\n" + \
                       "爆炸点高度:" + str(round(last_tar_z))
            ax.text(last_tar_x, last_tar_y, last_tar_z, axis_tar, color='blue')
            ax.scatter(last_tar_x, last_tar_y, last_tar_z, marker="v", c="blue")
        else:
            # 对交汇点设置文本标注信息
            axis_tar1 = "未炸毁目标，弹幕距离为" + str(round(last_pos_bao, 2))
            ax.text(last_tar_x, last_tar_y, last_tar_z, axis_tar1, color='blue')
            ax.scatter(last_tar_x, last_tar_y, last_tar_z, marker="v", c="blue")

        plt.legend()
        plt.show()


    # 二维绘图
    def poltting_2d(self,datasets,rows):
        if self.getvalue_x == '导弹经度' or self.getvalue_x == '导弹纬度' \
                or self.getvalue_x == '目标纬度' or self.getvalue_x == '目标经度':
            arr_x = datasets[self.getvalue_x].values
            xdata = np.linspace(arr_x.min(),arr_x.max(),rows)
        else:
            xdata = datasets[self.getvalue_x]
        if self.getvalue_y == '导弹经度' or self.getvalue_y == '导弹纬度' \
                or self.getvalue_y == '目标纬度' or self.getvalue_y == '目标经度':
            arr_y = datasets[self.getvalue_y].values
            ydata = np.linspace(arr_y.min(),arr_y.max(),rows)
        else:
            ydata = datasets[self.getvalue_y]

        plt.plot(xdata, ydata,color='red')
        plt.xlabel(self.getvalue_x)
        plt.ylabel(self.getvalue_y)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = new_summer()
    MainWindow.show()
    sys.exit(app.exec_())