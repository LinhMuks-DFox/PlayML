# Gradient Descent

目录：

* [梯度下降法介绍](#Introduction-Of-Gradient-Descent)
* [图像绘制代码](#Plot-code)



### <span id="Introduction-Of-Gradient-Descent">梯度下降法介绍</span>

* 不是一个机器学习算法
* 是一种基于搜索的最优化方案
* 作用：最小化一个损失函数
* 需要最大化效用函数时可以使用梯度上升法。

<p style="align:center"><img src="./pngs/GradientDescent_1.png" style="zoom:75%; "/></p>



当我们定义了一个损失函数$J$，y轴为损失函数$J$的取值，x轴为损失函数$J$的参数$\theta$的取值。

损失函数应该在一定范围内有一个最小值，最小化这个损失函数的过程来说，相当于在这个坐标系中找到一个点参数，使得损失函数$J$取得最小值。

在这个例子中，损失函数处于二位平面，相当于参数只有一个，并且我人为的指定其表达式为：
$$
J = 2 \theta^2 -4\theta + 2 \\
\frac{dJ}{d\theta} = 4\theta-4
$$
并且$\theta$的取值范围是$[0, 2)$

直线方程组，导数代表斜率，在曲线方程中，导数代表每个点的切线的斜率，换一个角度来理解，导数其实意味着，$\theta$单位变化时，$J$相应的变化。其实就是斜率的定义。

<p style="align:center"><img src="./pngs/GradientDescent_2.png" style="zoom:100%; "/></p>



* 橙色的点上，导数是一个负值，此时，$\theta$若减小（往左诺），$J$将变大（往上挪）
* 绿色的点上，导数为0，此时，$\theta$若减小，$J$也将减小
* 红色的点上，导数是一个正值，此时，$\theta$若减小（往左诺），$J$将变大（往下挪）

也就是说，导数其实也可以代表一个方向，对应$J$增大的方向

* 在橙色的点上，导数是负值，$J$增大的方向，是x轴的负方向
* 在红色的点上，导数是负值，$J$增大的方向，是x轴的正方向

如果想找到$J$的最小值，在任意点上，只需要取导数的相反数即可，搜索工程上对应也需要一个步长，移动多少总得确定下来，这个值一般我们用$\eta$来表示。

对于一个点来说：

* 求出其导数，就知道$J$增大的方向，要找出减少方向需要乘以$-1$
* 知道$J$减少的方向，需要确定一个步长，用于搜索。

$\eta$的值，不是固定的，但是一般取$0.1$。

下图中，从橙色的点，向$J$减少的方向前进一定的步数，就会抵达绿色的点

<p style="align:center"><img src="./pngs/GradientDescent_3.png" style="zoom:100%; "/></p>

只要步长合适，重复一定次数的前进，就可以抵达$J$的最小值，也就是$\theta=1.00$的时候



这个思想就是，**梯度下降法**，1维的函数的导数叫做导数，在多维函数中，要对各个方向的分量求导，最终得到的方向就是所谓的”梯度“。

另外，对于$\eta$

* 它被称作学习率(Learning Rate)
* 其取值将影响获得最优解的速度
* 取值不合适甚至无法得取最优解
* 他是梯度下降法的一个超参数

$\eta$太小：

<p style="align:center"><img src="./pngs/GradientDescent_4.png" style="zoom:50%; "/></p>

$\eta$太大：

<p style="align:center"><img src="./pngs/GradientDescent_5.png" style="zoom:50%; "/></p>

对于梯度下降法，在例子中是一个二次函数，它显然有唯一的极值点，但是并不是所有的函数都有唯一的极值点，很多时候，机器学习使用的梯度下降法面临的函数是很复杂的，比如：

<p style="align:center"><img src="./pngs/GradientDescent_6.png" style="zoom:50%; "/></p>

这个函数，有两个极小值，最小值在左边，使用梯度下降法进行搜索的话，初始点从右开始，搜索到的第一个”最小值“，显然不是这个函数的最小值，这种情况被叫做*局部最优解*，然而左边的极小值才是*全局最优解*。

解决方案：

* 多次运行，随机化初始点。
* 梯度下降法的初始点也是一个超参数。

详细内容参见[模拟梯度下降法](../notebooks/chp4-Gradient-Descent-And-Linear-Regression/01-Gradient-Descent-Simulations.ipynb)

### <span id="Plot-code">上述图片绘制代码</span>

```Python
# 上述图像的绘制代码
import matplotlib.pyplot as plt
import numpy as np
# 原函数
def J(thetas: np.ndarray):
    return 2 * thetas ** 2 - 4 * thetas + 2
# 函数导数
def dj(thetas: np.ndarray):
    return 4 * thetas - 4

# 绘制直线，point是在哪个点，k是斜率，X是取值范围
# 使用点斜式表达直线
def line(point, X, k):
    return k * (X - point[0]) + point[1]

# 橙，绿，红三点
dj_ls_0_point_on_J = Thetas[50], Y[50]
dj_eq_0_point_on_J = Thetas[100], Y[100]
dj_gt_0_point_on_J = Thetas[150], Y[150]

# 橙，绿，红三点的导数
dj_ls_0 = dj(dj_ls_0_point_on_J[0])
dj_eq_0 = dj(dj_eq_0_point_on_J[0])
dj_gt_0 = dj(dj_gt_0_point_on_J[0])

# 绘图
plt.plot(Thetas, Y)
plt.scatter(*dj_ls_0_point_on_J, color="orange")
plt.scatter(*dj_eq_0_point_on_J, color="g")
plt.scatter(*dj_gt_0_point_on_J, color="purple")

# 绘制橙，绿，红三点的切线
plt.plot(Thetas[25:75], line(dj_ls_0_point_on_J, Thetas[25:75], dj_ls_0))
plt.plot(Thetas[75:125], line(dj_eq_0_point_on_J, Thetas[75:125], dj_eq_0))
plt.plot(Thetas[125:175], line(dj_gt_0_point_on_J, Thetas[125:175], dj_gt_0))
# plt.plot(Thetas, dY, color="r")
plt.xlabel("Thetas")
plt.ylabel("Loss Function J Value")
plt.title("Gradient Descent Sample")
plt.show()
```