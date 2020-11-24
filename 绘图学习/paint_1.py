import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文使用
plt.rcParams['axes.unicode_minus'] = False # 用于显示负号
import numpy as np

'''
1 用subplot()方法绘制多幅图形
2 plt.axis([xmin, xmax, ymin, ymax]) 指定坐标x y的坐标范围
3 绘图： 点图(scatter)、柱状图(bar)、直线图(plot) 、饼图（pie）
4 设置图片标题 单独的图： plt.title() 多个图： axi.set_title()
5 实现不同图之间的切换
6 添加标签 plt.legend()
7 绘图时不同的颜色和表示形式（线段）：间scourses文件夹图片
'''


plt.figure(figsize=(6,6),dpi=80)
#创建第一个画板
plt.figure(1)
#将第一个画板划分为2行1列组成的区块，并获取到第一块区域
ax1 = plt.subplot(211)
#在第一个子区域中绘图
'''
设置颜色
'''
plt.scatter([1,3,5],[2,4,5],marker="v",s=50,color="r")#选中第二个子区域，并绘图
ax2 = plt.subplot(212)
plt.plot([2,4,6],[7,9,15])


#创建第二个画板
plt.figure(2)
x = np.arange(4)
y = np.array([15,20,18,25])
#在第二个画板上绘制柱状图
plt.bar(x,y)
#为柱状图添加标题
plt.title("第二个画板")


'''
饼图
'''
plt.figure(3)
labels = ['娱乐','育儿','饮食','房贷','交通','其它']
sizes = [2,5,12,70,2,9]
explode = (0,0,0,0.1,0,0)
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
plt.title("饼图示例-8月份家庭支出")

'''
实现标签显示
'''
plt.figure(4)
labels = 'A','B','C','D'
sizes = [10,10,10,70]
plt.pie(sizes,labels=labels)
plt.title("饼图详解示例")
plt.text(1,-1.2,'By:zzc') # 添加标注
# plt.legend(['a', 'b', 'c', 'd'], loc='upper right', fontsize=10)
plt.legend()

#切换到第一个画板
plt.figure(1)
#为第一个画板的第一个区域添加标题
ax1.set_title("第一个画板中第一个区域")
ax2.set_title("第一个画板中第二个区域")
# 调整每隔子图之间的距离

plt.tight_layout()
plt.show()