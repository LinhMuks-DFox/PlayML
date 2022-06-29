# 逻辑回归算法

* 逻辑回归：解决分类问题
* 将样本的特征和样本法身的概率联系起来，概率是一个数字，所以可以叫做回归问题

对于逻辑回归来说，$\hat{y} = f(x)$所获取到的预测值，本质上是一个概率，因此，这个简单的表达式也可以换成：
$$
\hat{p} = f(x)
$$
根据函数计算出的概率值，进行分类：
$$
\hat{p} = f(x) \\
\hat{y} = \left \{ 
\begin{align}
1, &\hat{p} \ge 0.5 \\
0, &\hat{p} \le 0.5
\end{align}
\right.
$$


因此，逻辑回归既可以是一个回归算法，也可以是一个分类算法，通常作为分类算法用，只可以解决2分类问题



在之前的章节中提到了线性回归算法，其最终求得到预测值的取值范围是正无穷到负无穷(=$(-\infin, \infin)$)，也就是说，可以计算出任意一个值

但是对于逻辑回归算法，最终求得的数值理应是一个概率，也就意味着这个值应该在 $[0, 1]$之间。

一个简单的解决办法就是，使用线性回归算法求出一个值，再把这个值放入函数$\sigma()$中，求出一个 $[0, 1]$之间的值。
$$
\hat{p} = \sigma(\theta^T \cdot x_b)
$$
这个sigma函数，一般采用**Sigmoid**函数，这个函数的表达式是：
$$
\sigma(t) = \frac{1}{1 + e^{-t}}
$$

其函数图像为：
<p style="align:center"><img src="./pngs/Logisitic-Regression_1.png" style="zoom:30%; "/></p>

Sigmoid函数的性质：

* 最左端逐渐趋近于0，但达不到0，最右端逐渐趋近于1，但达不到1，值域在0到1之间。
* 当$x = 0$的时候，函数的取值是0.5
* 当x小于0的时候，x越大，输出越趋近于0.5
* 当x大于0的时候，x越小，输出越趋近于0.5



综上，这个函数可以写成：
$$
\hat{p} = \sigma(\theta^T \cdot x_b) = \frac{1} {1 + e^{-\theta^{T} \cdot x_b}} \\

\hat{y} = \left \{ 
\begin{align}
1, &\hat{p} \ge 0.5 \\
0, &\hat{p} \le 0.5
\end{align}
\right.
$$
对应的问题来了：

**对于给定的样本数据集$(X, y)$，如何找到参数$\theta$，使得上述公式，可以最大程度的获得样本数据$X$对应的分类输出$y$？**

 

#### 逻辑回归的损失函数

在之前的表达式


$$
\hat{y} = \left \{ 
\begin{align}
1, &\hat{p} \ge 0.5 \\
0, &\hat{p} \le 0.5
\end{align}
\right.
$$
中，输出结果的时候被分为两部分，因此，我们也可以试着将损失函数$cost()$分成两部分：
$$
cost = \left \{ 
\begin{align}
  &y(真值) = 1,\text{ p越小，cost越大} \\ 
	&y(真值) = 0, \text{ p 越大，cost越大} 
\end{align}
\right.
$$

* 给定的样本的真值为1时，p越小（=越倾向于将样本分类为0），说明，sigmoid给出的偏离越大。

* 同理，给定的样本的真值为0时，p越大（=越倾向于将样本分类为1），说明，sigmoid给出的偏离越大。

根据上文描述，可以轻松的想到，可以使用这两个函数来表示损失函数：
$$
cost = \left \{
  \begin{align}
  	-\log(\hat{p}) \space, \text{if} \space y = 1 \\
  	-\log(\hat{p} - 1) \space, \text{if} \space y = 0
  \end{align}
\right.
$$
log函数的曲线：
<p style="align:center"><img src="./pngs/Logisitic-Regression_2.png" style="zoom:30%; "/></p>

log函数经过(1, 0)这个点，$-\log{(x)}$这个函数的图像是这样的：

<p style="align:center"><img src="./pngs/Logisitic-Regression_3.png" style="zoom:30%; "/></p>

<p style="align:center"><img src="./pngs/Logisitic-Regression_4.png" style="zoom:30%; "/></p>

但是，注意这里传入的参数是$\hat{p}$，他的取值范围是0到1，y=1时的损失函数的图像为：

<p style="align:center"><img src="./pngs/Logisitic-Regression_5.png" style="zoom:30%; "/></p>

可以观察发现：

* $\hat{p}$趋近于0的时候，损失函数输出的值趋近于正无穷，也就是$\hat{p}$越小，cost越大



同理，考虑$\hat{p}$的取值范围，$-\log(\hat{p} - 1)$的图像为：

<p style="align:center"><img src="./pngs/Logisitic-Regression_6.png" style="zoom:30%; "/></p>

* $\hat{p}$趋近于1的时候，损失函数输出的值趋近于正无穷，也就是$\hat{p}$越大，cost越大

综上所述，
$$
cost = \left \{
  \begin{align}
  	-\log(\hat{p}) \space, \text{if} \space y = 1 \\
  	-\log(\hat{p} - 1) \space, \text{if} \space y = 0
  \end{align}
\right.
$$
满足给定的条件：

* 真值为1时，p越小（=越倾向于将样本分类为0），说明，sigmoid给出的偏离越大。

* 真值为0时，p越大（=越倾向于将样本分类为1），说明，sigmoid给出的偏离越大。
