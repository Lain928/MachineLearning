import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ui.untitled as tt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
import math


class Win(QWidget,tt.Ui_Form):

    def __init__(self):
        super().__init__()
        self.lla_x = []
        self.lla_y = []
        self.lla_z = []

        self.setupUi(self)
        self.filepath = ''
        # self.radioLLA.setChecked(True) # 默认是lla形式
        # self.radioECEF.setChecked(False)
        self.pushButton.clicked.connect(self.getfilepath)
        self.pushButton_2.clicked.connect(self.paint_3d)
        self.pushButton_3.clicked.connect(self.close)

        # 下拉框
        self.getvalue_x = '导弹经度'
        self.getvalue_y = '导弹纬度'
        self.getvalue_z = '导弹高度'
        self.comboBox_x.addItems(['导弹经度','导弹纬度','导弹高度','导弹距离','导弹速度','导弹过载','目标高度','弹目距离'])
        self.comboBox_y.addItems(['导弹经度','导弹纬度','导弹高度','导弹距离','导弹速度','导弹过载','目标高度','弹目距离'])
        self.comboBox_z.addItems(['导弹经度','导弹纬度','导弹高度','导弹距离','导弹速度','导弹过载','目标高度','弹目距离'])
        # 下拉框默认选项
        self.comboBox_x.setCurrentIndex(0)  # 设置默认值
        self.comboBox_y.setCurrentIndex(1)  # 设置默认值
        self.comboBox_z.setCurrentIndex(2)  # 设置默认值
        # 信号
        # x
        self.comboBox_x.currentIndexChanged[str].connect(self.print_value_x) # 条目发生改变，发射信号，传递条目内容
        # self.comboBox_x.currentIndexChanged[int].connect(self.print_value_x)  # 条目发生改变，发射信号，传递条目索引
        # y
        self.comboBox_y.currentIndexChanged[str].connect(self.print_value_y) # 条目发生改变，发射信号，传递条目内容
        # self.comboBox_y.currentIndexChanged[int].connect(self.print_value_y)  # 条目发生改变，发射信号，传递条目索引
        #z
        self.comboBox_z.currentIndexChanged[str].connect(self.print_value_z) # 条目发生改变，发射信号，传递条目内容
        # self.comboBox_z.currentIndexChanged[int].connect(self.print_value_z)  # 条目发生改变，发射信号，传递条目索引


    def getfilepath(self):
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        self.lineEdit.setText(directory[0])


    def paint_3d(self):
        if self.filepath == '':
            return "请先选择处理文件！！！"

        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        print(datasets.head())
        #datasets.columns = ['X', 'Y', 'Z']

        # if self.radioLLA.isChecked()==True:
        xdata = datasets[self.getvalue_x]
        ydata = datasets[self.getvalue_y]
        zdata = datasets[self.getvalue_z]
        # if self.radioECEF.isChecked()==True:
        #     xdata,ydata,zdata = self.ecef_lla(datasets)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata)
        ax.set_xlabel(self.getvalue_x)
        ax.set_ylabel(self.getvalue_y)
        ax.set_zlabel(self.getvalue_z)
        plt.show()


    def print_value_x(self, i):
        self.getvalue_x = i
    def print_value_y(self, i):
        self.getvalue_y = i
    def print_value_z(self, i):
        self.getvalue_z = i


if __name__=='__main__':
    app=QApplication(sys.argv)
    w=Win()
    w.show()
    sys.exit(app.exec_())
