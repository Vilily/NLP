# Reasoning about Entailment with Neural Attention

![image-20201008231425441](C:\Users\WILL\AppData\Roaming\Typora\typora-user-images\image-20201008231425441.png)

* **输入**：$[x^{pre}_1, x^{pre}_2,\cdots, x^{pre}_n] : x^{pre}_t\in \mathbb{R}^{embedding\_size}$，输入的**前提**词向量；$[x^{hyp}_1, x^{hyp}_2,\cdots, x^{hyp}_m]x:^{hyp}_t\in \mathbb{R}^{embedding\_size}$，输入的**推断**词向量；
* **输入**：输入词向量一般用**word2vec**，**out-of-vocabulary**用**(-0.05,0.05)正态分布随机初始化**。
* **LSTM**层：$[h_1^{pre},h_2^{pre},\cdots,h_n^{pre}]=LSTM\_A(x_t^{pre})$
* **LSTM**层：$[h_1^{hyp},h_2^{hyp},\cdots,h_n^{hyp}]=LSTM\_A(x_t^{hyp})$
* **attention**：
* **word2word-attention**：