# GloVe

### 简介

**Skip-gram**和**CBOW**都是基于**窗口**的模型，无法获取全篇文章的信息，因此具有局限性。

### 模型公式

***

$$F(\pmb{w}_i,\pmb{w}_j,\pmb{\tilde{w}}_k)=\frac{P_{ik}}{P_{jk}}$$

**其中**

$\pmb{\tilde{w}}_k\in \mathbb{R}^d$：上下文单词的编码

$\pmb{w}_i\in\mathbb{R}^d$：文本单词的**编码**

$P_{ij}$：是单词$j$是单词$i$的**上下文**的概率，即：$P_{ij}=\frac{N_{ij}}{Ni}$

$N_{ij}$：是在存在单词$i$为**上下文**的情况下，单词$j$出现的次数；

$N_i=\sum_kN_{ik}$：是单词$i$出现的次数

$F(\cdot)$：某个函数

***

因为单词的**向量空间**是线性的，所以：

$$F(w_i-w_j,\tilde w_k)=\frac{P_{ik}}{P_{jk}}$$												<font color=red>（1）</font>

***

注意到：<font color=red>(1)</font>中$F(\cdot)$参数为**向量**，但其右边为**标量**，所以：

$$F((w_i-w_j)^T\tilde w_k)=\frac{P_{ik}}{P_{jk}}$$											<font color=red>（2）</font>

***

注意到：一个**中心词**和**上下文词**是任意的，可以互换的，所以：

$$F((w_i-w_j)^T\tilde w_k)=\frac{F(w_i^T\tilde w_k)}{F(w_j^T\tilde w_k)}$$									<font color=red>（3）</font>

**其中**：

$F(w_i^T\tilde w_k)=P_{ik}=\frac{X_{ik}}{Xi}$

***

<font color=red>（3）</font>式的**解**为：

$F=exp(\cdot)$或者$w_i^T\tilde w_k=\log{P_{ik}}=\log{X_{ik}}-\log{X_{i}}$		<font color=red>（4）</font>

因为$X_i$独立于$k$，所以：

$$w_i^T\tilde w_k+b_i+\tilde b_k=\log(X_{ik})$$										<font color=red>（5）</font>

***

**代价函数**

$$J=\sum_{i,j=1}^Vf(X_{ij})(w_i^T\tilde w_j+b_i+\tilde b_j-\log{X_{ij}})^2$$

**其中**

$$f(x)=\left\{ \begin{array}{**lr**} (x/x_{max})^\alpha \quad if x<x_{max} \\ \qquad 1 \qquad otherwise \end{array} \right.$$

$\alpha=\frac3 4$