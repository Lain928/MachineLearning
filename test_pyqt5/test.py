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
        self.radioBtnThree.setChecked(True) # 默认是绘制三维图
        self.radioBtnTwo.setChecked(False)
        self.pushButton.clicked.connect(self.getfilepath)
        self.pushButton_2.clicked.connect(self.mypaint)
        self.pushButton_3.clicked.connect(self.close)

        # # 删除选项
        # self.radioBtnTwo.toggled.connect(self.delete_z)

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


    # def delete_z(self):
    #     self.horizontalLayout_4.itemAt(4).widget().deleteLater()
    #     self.horizontalLayout_4.itemAt(5).widget().deleteLater()
    # def add_z(self):
    #     btncont= self.layout.count()
    #     widget = QPushButton(str(btncont-1), self)
    #     self.layout.addWidget(widget)


    def getfilepath(self):
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        self.lineEdit.setText(directory[0])


    def mypaint(self):
        if self.filepath == '':
            return "请先选择处理文件！！！"

        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        rows = datasets.shape[0]
        xdata = datasets[self.getvalue_x]
        ydata = datasets[self.getvalue_y]
        zdata = datasets[self.getvalue_z]

        if self.radioBtnTwo.isChecked()==True:
            plt.plot(xdata, ydata)
            plt.xlabel(self.getvalue_x)
            plt.ylabel(self.getvalue_y)
            plt.grid()
            plt.show()
        if self.radioBtnThree.isChecked()==True:
            self.paltting_3d(xdata,ydata,zdata)


    def print_value_x(self, i):
        self.getvalue_x = i
        print(i)
    def print_value_y(self, i):
        self.getvalue_y = i
        print(i)
    def print_value_z(self, i):
        self.getvalue_z = i
        print(i)


    def paltting_3d(self,xdata,ydata,zdata):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata)
        ax.set_xlabel(self.getvalue_x)
        ax.set_ylabel(self.getvalue_y)
        ax.set_zlabel(self.getvalue_z)
        plt.show()
        pass
    def platting_double(self):
        datasets = pd.read_table(self.filepath, sep=',',encoding='gb18030')
        rows = datasets.shape[0]
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
        ax.set_title("导弹弹道数据分析")

        # 对交汇点设置文本标注信息
        axis_mis = "mis_lon:" + str(round(last_mis_x / 3.14 * 180, 2)) + "\n" \
                   + "mis_lat:" + str(round(last_mis_y / 3.14 * 180, 2)) \
                   + "\n" + "mis_alt:" + str(round(last_mis_z))
        ax.text(last_mis_x, last_mis_y, last_mis_z + 2000.0, axis_mis, color='blue')
        ax.scatter(last_mis_x, last_tar_y, last_tar_z, marker="v", c="blue")

        axis_tar = "tar_lon:" + str(round(last_tar_x / 3.14 * 180, 2)) + "\n" \
                   + "tar_lat:" + str(round(last_tar_y / 3.14 * 180, 2)) \
                   + "\n" + "tar_alt:" + str(round(last_tar_z))
        ax.text(last_tar_x, last_tar_y, last_tar_z - 6000.0, axis_tar, color='blue')
        ax.scatter(last_tar_x, last_tar_y, last_tar_z, marker="v", c="blue")
        plt.legend()
        plt.show()

if __name__=='__main__':
    app=QApplication(sys.argv)
    w=Win()
    w.show()
    sys.exit(app.exec_())
