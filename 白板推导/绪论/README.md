# 绪论

$$X=\begin{pmatrix} x_1 & x_2 & x_3 & \cdots & x_n\end{pmatrix}^T_{N\times P}$$

$$=\begin{pmatrix}x_{11} & x_{12} & \cdots & x_{1P} \\ x_{21} & x_{22} & \cdots & x_{2P} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N1} & x_{N2} & \cdots & x_{NP}\end{pmatrix}$$

$$\theta$$：模型参数

$$x\sim p(x|\theta)$$

## 频率派（统计机器学习）

观点：$\theta$是未知**常量**

$$\theta_{MLE}=argmax_{\theta} logP(x|\theta)$$



## 贝叶斯派（概率图模型）

观点：$\theta$服从某一**分布**，$\theta\sim p(\theta)$

$$p(\theta|X)=\frac{p(X|\theta)\cdot p(\theta)}{p(X)}$$

**因为**$p(X)$**不变**

$$\theta_{MAP}=argmax_{\theta}p(\theta|X)=argmax_{\theta}p(X|\theta)\cdot p(\theta)$$

