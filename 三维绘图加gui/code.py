from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


frameT = Tk()
frameT.geometry('500x200+400+200')
frameT.title('选择需要输入处理的文件')
frame = Frame(frameT)
frame.pack(padx=10, pady=10)  # 设置外边距
frame1 = Frame(frameT)
frame1.pack(padx=10, pady=10)
v1 = StringVar()
#ent = Entry(frame, width=50, textvariable=v1).pack(fill=X, side=LEFT)  # x方向填充,靠左


def fileopen():
    file_sql = askopenfilename()
    # 将路径输出到文本框中
    if file_sql:
        v1.set(file_sql)


def paint_3d(path):
    datasets = pd.read_csv(path, sep=' ', header=None)
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


def match():
    paint_3d(v1.get())
    pass


btn = Button(frame, width=20, text='选择文件', font=("宋体", 14), command=fileopen).pack(fil=X, padx=10)
ent = Entry(frame, width=50, textvariable=v1).pack(fill=X, side=LEFT)  # x方向填充,靠左
ext = Button(frame1, width=10, text='运行', font=("宋体", 14), command=match).pack(fill=X, side=LEFT)
etb = Button(frame1, width=10, text='退出', font=("宋体", 14), command=frameT.quit).pack(fill=Y, padx=10)
frameT.mainloop()