import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
import math


datasets = pd.read_table("C:/Users/zzc/Desktop/200/11.csv", sep=',',encoding='gb18030')
print(datasets)


# ecef坐标转经纬高
def ecef_lla(self, datasets):
    for i in range(datasets.shape[0]):
        xdata = datasets.loc[i, 0]
        ydata = datasets.loc[i, 1]

        zdata = datasets.loc[i, 2]

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
