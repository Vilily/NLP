# Enhanced LSTM for Natural Language Inference

## ESIM

![image-20201016184513863](.\readme_pic\image-20201016184513863.png)

**input**：

* $a=(a_1,a_2,\cdots,a_{l_a}),a_i\in \mathbb{R}^l$：**前提**；
* $b=(b_1,b_2,\cdots,b_{l_b}),b_i\in \mathbb{R}^l$：**假设**；
* 词向量可以用预训练的词嵌入模型，也可以用语法分析树。

**output**：

* $y$：**label**，表示**前提**和**假设**之间的关系。

### 1. Input Encoding

使用**BiLSTM**作为基本单元。

* $\bar{a_i}=BiLSTM(a, i),\forall i\in\{1,\cdots,l_a\}$；将$a$送入**LSTM**，得到**hidden state** $\bar{a}$；
* $\bar{b_i}=BiLSTM(b, i),\forall i\in\{1,\cdots,l_b\}$；将$b$送入**LSTM**，得到**hidden state** $\bar{b}$；
* $a、b$分别进行$LSTM$
* **BiLSTM**的**hidden state**为前后向的**hidden state**叠加；



### 2. Local Inference Modeling

* $e_{ij}=\bar{a}_i^T\bar{b}_j.$：获取**前提**和**假设**的关系
* $\tilde{a}_i=\sum_{j=1}^{l_b}=\frac{\exp(e_{ij})}{\sum_{k=1}^{l_b}\exp(e_{ik})}\bar{b}_j,\forall i\in\{1, 2, \cdots, l_a\}$：
* $\tilde{b}_j=\sum_{i=1}^{l_a}\frac{\exp{e_{ij}}}{\sum_{k=1}^{l_a}\exp(e_{kj})}\bar{a}_i,\forall j\in\{1,\cdots,l_b\}$：
* $\bold{m}_a=[\bar{a};\tilde{a};\bar{a}-\tilde{a};\bar{a}\odot \tilde{a}]$：
* $\bold{m}_b=[\bar{b};\tilde{b};\bar{b}-\tilde{b};\bar{b}\odot\tilde{b}]$：

* $m_{ai}\in\mathbb{R}^{4h}$
* $m_{bj}\in\mathbb{R}^{4h}$

### 3. Inference Composition

* $v_a=BiLSTM(m_a, i),\forall i\in\{1,\cdots,l_a\}$；将$m_a$送入**LSTM**，得到**hidden state** $v_a$；
* $v_b=BiLSTM(m_b, i),\forall i\in\{1,\cdots,l_b\}$；将$m_b$送入**LSTM**，得到**hidden state** $v_b$；

为增强鲁棒性，放弃直接**求和**，使用**求平均和最大值**；

* $\bold{v}_{a,ave}=\sum_{i=1}^{l_a}\frac{\bold{v}_{a,i}}{l_a},\quad \bold{v}_{a.max}=\max_{i=1}^{l_a}\bold{v}_{a,i}$
* $\bold{v}_{b,ave}=\sum_{i=1}^{l_b}\frac{\bold{v}_{b,i}}{l_b},\quad \bold{v}_{b.max}=\max_{i=1}^{l_b}\bold{v}_{b,i}$
* $\bold{v}=[\bold{v}_{a,ave};\bold{v}_{a,max};\bold{v}_{b,ave};\bold{v}_{b, max}]$



### 4. Calssify

* **tanh**
* **MLP**
* **softmax**
* **cross-entropy**

