# End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF

## Network

### CNN for Character-level Representation



<img src="D:\NLP\NLP\BiLSTM-CNNs-CRF\md_pic\CNN.png" alt="image-20201022200229254" style="zoom:80%;" />

* **INPUT**：$bold{I}\in\mathbb{R}^{length\times embedding\_size}$：一个句子每个单词*Embed*成一个*hidden*；

* **dropout**：在**character-embedding**送入**CNN**之前；
* **Conv**：**Kernel_Size**：$n\times1$，**Padding_Size**：$k$；$\mathbb{R}^{length\times embedding\_size}\to \mathbb{R}^{length+2*k-n+1\times embedding\_size}$
* **Max_Pooling**：对*length*进行*pooling*，$\mathbb{R}^{length+2*k-n+1\times embeddig\_size}\to \mathbb{R}^{embedding\_size}$
* **OUTPUT**：$\bold{O}\in\mathbb{R}^{embedding\_size}$



### Bi-directional LSTM

