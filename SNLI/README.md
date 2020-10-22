# Reasoning about Entailment with Neural Attention

### Network

![image-20201008231425441](C:\Users\WILL\AppData\Roaming\Typora\typora-user-images\image-20201008231425441.png)

* **输入**：$[x^{pre}_1, x^{pre}_2,\cdots, x^{pre}_n] : x^{pre}_t\in \mathbb{R}^{embedding\_size}$，输入的**前提**词向量；$[x^{hyp}_1, x^{hyp}_2,\cdots, x^{hyp}_m]x:^{hyp}_t\in \mathbb{R}^{embedding\_size}$，输入的**推断**词向量；

* **输入**：输入词向量一般用**word2vec**，**out-of-vocabulary**用**(-0.05,0.05)正态分布随机初始化**。

* **参数**：

  $\bold{W}_p,\bold{W}_x\in\mathbb{R}^{hidden\_size\times hidden\_size}$

* **前向传播**：

* **LSTM**层：$[h_1^{pre},h_2^{pre},\cdots,h_n^{pre}]=LSTM\_A(x_t^{pre})$

* **LSTM**层：$[h_1^{hyp},h_2^{hyp},\cdots,h_n^{hyp}]=LSTM\_A(x_t^{hyp})$

* **attention**：$\bold{r}=attention([h^{pre}_i], h^{hyp}_n)$

* **wordBYword-attention**：$\bold{r}=wordBYword-attention([h^{pre}_i],[h^{hyp}_i])$

* $\bold{h}^*=\tanh(\bold{W}^p\bold{r}+\bold{W}^x\bold{h}_N)$



### Attention

* **输入**：$Y\in\mathbb{R}^{hidden\_size\times pre\_length}$：由**Pre_LSTM**输出向量$[h_1,\cdots,h_t]$组成的；

  $\bold{e_L}\in\mathbb{R}^L$：是由**1**组成的向量；

  $\bold{h}_N\in\mathbb{R}^{hidden\_size}$：**Hyp_LSTM**的最终输出向量；

* **参数**：

  $\bold{W}_y,\bold{W}^h\in\mathbb{R}^{hidden\_size\times hidden\_size}$

  $\bold{w}\in\mathbb{R}^{hidden\_size}$

* **前向传播**：

* $\bold{M}=\tanh(\bold{W^y}\bold{Y}+\bold{W}^h\bold{h}_N\otimes\bold{e}_L)$
* $\alpha=softmax(\bold{w}^T\bold{M})$
* $\bold{r}=\bold{Y}\alpha^T$



### Word-by-word Attention

* **输入**：

  $Y\in\mathbb{R}^{hidden\_size\times pre\_length}$：由**Pre_LSTM**输出向量$[h_1,\cdots,h_t]$组成的；

  $\bold{e_L}\in\mathbb{R}^L$：是由**1**组成的向量；

  $\bold{h}_N\in\mathbb{R}^{hidden\_size}$：**Hyp_LSTM**的最终输出向量；

* **参数**：

  $\bold{W}_y,\bold{W}^h\in\mathbb{R}^{hidden\_size\times hidden\_size}$

  $\bold{w}\in\mathbb{R}^{hidden\_size}$

* **前向传播**：

* $\bold{M}_t=\tanh(\bold{W^y}\bold{Y}+(\bold{W}^h\bold{h}_t+\bold{W}^r\bold{r}_{t-1})\otimes\bold{e}_L)$
* $\alpha=softmax(\bold{w}^T\bold{M}_t)$
* $\bold{r}=\bold{Y}\alpha^T+\tanh(\bold{W}^t\bold{r}_{t-1})$

