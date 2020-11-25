import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import leastsq
from scipy import constants as C


u1 = np.array([0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46])
u2 = np.array([0.026,0.038,0.058,0.084,0.121,0.185,0.28,0.416,0.597,0.917,1.341,2.07,3.005,4.379,6.575,9.523,12.001])
lnu2 = np.log(u2)

# 误差函数
def residuals(p):
    k, b = p
    return lnu2 - (k * u1 + b)


r = leastsq(residuals, [1, 0])
K, B = r[0]
print("斜率k =", K, "截距b =", B)
a, b = np.exp(B), K
print("常量a =", a, "常量b =", b)
k = C.electron_volt / (b * 300)
print("玻尔兹曼常量k =", k, "J/K")




plt.scatter(u1, lnu2)
plt.plot(u1, u1 * K + B, linewidth=2)
plt.xlabel('U1(V)')
plt.ylabel('lnU2(V)')
plt.grid(True)
plt.title('Plot : lnU2--U1')
plt.show()

plt.scatter(u1, u2)
plt.plot(u1, a * np.exp(b * u1), linewidth=2)
plt.xlabel('U1(V)')
plt.ylabel('U2(V)')
plt.grid(True)
plt.title('Plot : U2--U1')
plt.show()