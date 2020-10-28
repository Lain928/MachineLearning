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
        self.radioLLA.setChecked(True) # 默认是lla形式
        self.radioECEF.setChecked(False)
        self.pushButton.clicked.connect(self.getfilepath)
        self.pushButton_2.clicked.connect(self.paint_3d)
        self.pushButton_3.clicked.connect(self.close)


    def getfilepath(self):
        # 获取文件
        directory = QFileDialog.getOpenFileName(None, "选取文件", "./", "All Files (*);;Text Files (*.txt)")
        self.filepath = directory[0]
        self.lineEdit.setText(directory[0])


    def ecef_lla(self,datasets):
        for i in range(datasets.shape[0]):
            xdata = datasets.loc[i, 'X']
            ydata = datasets.loc[i, 'Y']
            zdata = datasets.loc[i, 'Z']

            F = 1 / 298.257
            L_EQUATORA = 6378140.0
            B = 0.1
            ep = 1e-25
            e2 = 2 * F - F * F

            resultx = math.atan2(ydata, xdata)
            B1 = math.atan2(zdata, math.sqrt(xdata * xdata + ydata * ydata))
            while 1:
                N1 = L_EQUATORA / math.sqrt(1 - e2 * math.sin(B1) * math.sin(B1))
                B = math.atan2(zdata + N1 * e2 * math.sin(B1), math.sqrt(xdata * xdata + ydata * ydata))
                if math.fabs(B - B1) < ep:
                    break
                else:
                    B1 = B

            resulty = B
            N = L_EQUATORA / math.sqrt(1 - e2 * math.sin(B) * math.sin(B))
            resultz = math.sqrt(xdata * xdata + ydata * ydata) / math.cos(B) - N

            self.lla_x.append(resultx)
            self.lla_y.append(resulty)
            self.lla_z.append(resultz)
        return self.lla_x, self.lla_y, self.lla_z


    def paint_3d(self):
        if self.filepath == '':
            return "请先选择处理文件！！！"

        datasets = pd.read_csv(self.filepath, sep=' ', header=None)
        datasets.columns = ['X', 'Y', 'Z']

        if self.radioLLA.isChecked()==True:
            xdata = datasets.loc[:, 'X']
            ydata = datasets.loc[:, 'Y']
            zdata = datasets.loc[:, 'Z']
        if self.radioECEF.isChecked()==True:
            xdata,ydata,zdata = self.ecef_lla(datasets)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xdata, ydata, zdata)
        ax.set_xlabel('经度')
        ax.set_ylabel('维度')
        ax.set_zlabel('高度')
        plt.show()

if __name__=='__main__':
    app=QApplication(sys.argv)
    w=Win()
    w.show()
    sys.exit(app.exec_())
