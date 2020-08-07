""" GloVe算法的pytorch实现
@Author: Bao Wenjie
@Date: 2020/8/6
@Email: bwj_678@qq.com
"""

import torch

class Glove:
    def __init__(self, device, n, dim, batch_size):
        """ 初始化
        @param device: 训练设备
        @param n: 词典大小
        @param dim: 词向量维度
        """
        super().__init__()
        self.device = device
        self.dim = dim
        self.n = n
        self.batch_size = batch_size
        self.x_max = 100
        self.alpha = 3/4
        # 中心词向量
        self.vector = torch.autograd.Variable(torch.randn((n, dim),  requires_grad=True, device=device))
        self.b = torch.randn((n, 1), requires_grad=False, device=device)
        # 上下文词向量
        self.vector_ = torch.autograd.Variable(torch.randn((n, dim), requires_grad=True, device=device))
        self.b_ = torch.randn((1, n), requires_grad=False, device=device)

    
    def train(self, X):
        """ 训练过程
        @param X numpy(n, n): 共现矩阵
        """
        X = torch.from_numpy(X).to(dtype=torch.float32, device=self.device)
        # (n, n)
        L = torch.mm(self.vector, torch.transpose(self.vector_)) + self.b + self.b_ -torch.log(X)
        L = torch.pow(L, 2)
        # (n, n)
        F = X / x_max
        F = torch.pow(F, self.alpha)
        F[F >= 1] = 1
        # loss
        loss = torch.sum(torch.mul(F, L))
        return loss
    
    def embed(self, X):
        """ 词嵌入
        @param X numpy(n): 待嵌入的单词
        """
        return self.vector[X].numpy()
    
    def predict(self, X, top):
        """ 预测与单词X相关的单词
        """
        pass

    def load_model(self, path):
        """ 加载模型
        """ 
        pass

    def save_model(self, path):
        """ 保存模型
        """
        pass
