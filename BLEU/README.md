# BLEU模型

## 公式

$$p_n=\frac{\sum_{C\in\{Candidates\}}\sum_{n-gram\in C}Count_{clip}(n-gram)}{\sum_{C'\in\{Candidates\}}\sum_{n-gram'\in C'}Count(n-gram')}$$

**其中**

$Candidates$：表示**机器翻译的译文**

$Count()$：在**Candidates**中**n-gram**出现的次数

$Count_{clip}()$：**Candidates**中**n-gram**在**Reference**中出现的次数

****

$$BP=\left\{ \begin{aligned}1 \qquad if\quad c>r \\ e^{1-\frac{r}{c}} \qquad if \quad c\leq r \end{aligned}\right.$$

$c$：**Cadiddate**的长度

$r$：**Reference**的长度

****

$$BLEU=BP\cdot exp(\sum_{n=1}^Nw_n logp_n)$$

$w_n$：权重，一般为$\frac1N$

