# 数学基础

## 高斯分布

**DATA**

$$X=\begin{pmatrix} x_1 & x_2 & \cdots & x_N\end{pmatrix}^T=\begin{pmatrix}x_1^T \\ x_2^T \\ \vdots \\ x_N^T\end{pmatrix}_{N\times P}$$

$$x_i\in \mathbb{R}^P $$

$$x_i\sim_{lid}N(\mu, \Sigma)$$

$$\theta=(\mu,\Sigma)$$

***

#### 推导1

**目标**：$MLE:\theta_{MLE}$

**过程**：

$$P(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

$$P(x)=\frac{1}{\sqrt{2\pi|\Sigma|}}exp(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu))$$



$$logP(X|\theta)=log\prod_{i=1}^NP(x_i|\theta)=\sum_{i=1}^NlogP(x_i|\theta)$$

$$=\sum_{i=1}^Nlog\frac1{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

$$=\sum_{i=1}^N(log\frac1{\sqrt{2\pi}}+log\frac1{\sigma}-\frac{(x-\mu)^2}{2\sigma^2})$$



$$\mu_{MLE}=argmax_{\mu}logP(X|\theta)$$

$$=argmax(\sum_{i=1}^N-\frac{(x-\mu)^2}{2\sigma^2})$$

$$=argmin_{\mu}(\sum_{i=1}^Nx_i-\mu^2)$$



**求偏导，取极值**

$$\frac{\partial}{\partial\mu}\sum(x_i-\mu)^2=\sum_{i=1}^N2\cdot (x_i-\mu)\cdot -1=0$$

$$\sum_{i=1}^N(x_i-\mu)=0$$

$$\sum_{i=1}^Nx_i-\sum_{i=1}^N\mu=0$$

$$\mu_{MLE}=\frac1N\sum_{i=1}^Nx_i$$