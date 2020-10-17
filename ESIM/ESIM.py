# paper: Enhanced LSTM for Natural Language Inference
#        https://arxiv.org/abs/1609.06038
# author: bao wenjie
# email: bwwj_678@qq.com
# date: 2020/10/17

import torch
import torch.nn as nn
from vocab import Vocab

class ESIMModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, device, class_num, vocab:Vocab, batch_size=1):
        """ 初始化
        :param hidden_size: lstm隐藏层大小
        :param embedding_size: 词嵌入向量大小
        :param device: 训练设备
        :param class_num: 分类数量
        """
        super(ESIMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device
        self.class_num = class_num
        self.vocab = vocab
        # embedding model
        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=embedding_size)
        # Input Encoding
        self.pre_lstm = nn.LSTM(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                bias=True,
                                batch_first=True,
                                bidirectional=True)
        self.hyp_lstm = nn.LSTM(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                bias=True,
                                batch_first=True,
                                bidirectional=True)
        # Local Inference Modeling
        # Inference Composition
        self.infer_pre_lstm = nn.LSTM(input_size=8*hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      bias=True,
                                      batch_first=True,
                                      bidirectional=True)
        self.infer_hyp_lstm = nn.LSTM(input_size=8*hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      bias=True,
                                      batch_first=True,
                                      bidirectional=True)
        self.ave_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        # Classify
        self.tanh = nn.Tanh()
        self.MLP = nn.Linear(in_features=8*hidden_size,
                             out_features=class_num)
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def get_hidden_state0(self, batch_size):
        """ 获取h_0, c_0
        :return (h_0, c_0): h_0:(batch_size, hidden_size)
        """
        return (torch.zeros(size=(batch_size, self.hidden_size), device=self.device, requires_grad=False),
                torch.zeros(size=(batch_size, self.hidden_size), device=self.device, requires_grad=False))

    def forward(self, data, target=None):
        """ 前向传播
        :data (pre_data, hyp_data):训练数据,pre_data:(batch_size, pre_length), 
                                           hyp_data:(batch_size, hyp_length)
        :target :目标label, (batch_size)
        """
        pre_data, hyp_data = data
        pre_data = pre_data.long().to(self.device) # (batch_size, pre_length)
        hyp_data = hyp_data.long().to(self.device) # (batch_size, hyp_length)
        # Embedding 
        pre_embedding = self.embedding(pre_data) # (batch_size, pre_length, embedding_size)
        hyp_embedding = self.embedding(hyp_data) # (batch_size, hyp_length, embedding_size)
        # Input Encoding
        pre_hidden, _ = self.pre_lstm(pre_embedding) # (batch_size, pre_length, 2*hidden_size)
        hyp_hidden, _ = self.hyp_lstm(hyp_embedding) # (batch_size, hyp_length, 2*hidden_size)
        # Local Inference Modeling
        E = torch.matmul(pre_hidden, hyp_hidden.permute([0, 2, 1])) # (batch_size, pre_lengthm hyp_length)
        E = torch.exp(E)
        E_hyp = torch.mul(E.unsqueeze(dim=3), torch.unsqueeze(hyp_hidden, dim=2)) # (batch_size, pre_length, hyp_length, 2*hidden_size)
        E_pre = torch.mul(E.permute([0, 2, 1]).unsqueeze(dim=3), torch.unsqueeze(pre_hidden, dim=2)) # (batch_size, hyp_length, pre_length, 2*hidden_size)
        a_tilde = torch.sum(E_hyp, dim=2) / torch.sum(E, dim=2).unsqueeze(dim=2) # (batch_size, pre_length, 2*hidden_size)
        b_tilde = torch.sum(E_pre, dim=2) / torch.sum(E, dim=1).unsqueeze(dim=2) # (batch_size, hyp_length, 2*hidden_size)
        a_third = pre_hidden - a_tilde
        b_third = hyp_hidden - b_tilde
        a_forth = torch.mul(pre_hidden, a_tilde)
        b_forth = torch.mul(hyp_hidden, b_tilde)
        m_a = torch.cat((pre_hidden, a_tilde, a_third, a_forth), dim=2) # (batch_size, pre_length, 4*hidden_size)
        m_b = torch.cat((hyp_hidden, b_tilde, b_third, b_forth), dim=2) # (barch_size, hyp_length, 4*hidden_size)
        # Inference Composition
        v_pre, _ = self.infer_pre_lstm(m_a) # (batch_size, pre_length, hidden_size)
        v_hyp, _ = self.infer_hyp_lstm(m_b) # (batch_size, hyp_length, hidden_size)
        v_pre = v_pre.permute([0, 2, 1]) # (batch_size, hidden_size, pre_length)
        v_hyp = v_hyp.permute([0, 2, 1]) # (batch_size, hidden_size, hyp_length)
        v_pre_ave = self.ave_pooling(v_pre) # (batch, hidden_size)
        v_hyp_ave = self.ave_pooling(v_hyp) # ...
        v_pre_max = self.max_pooling(v_pre) # ...
        v_hyp_max = self.max_pooling(v_hyp) # ...
        v = torch.cat((v_pre_ave, v_pre_max, v_hyp_ave, v_pre_max), dim=1) # (batch_size, 4*hidden_size, 1)
        v = v.squeeze(dim=2)
        # Classify
        v_1 = self.tanh(v)
        v_2 = self.MLP(v_1) # (batch_size, class_num)
        # Output
        if target is None:
            # Predict
            y = self.softmax(v_2) # (batch_size, class_num)
            y = torch.argmax(y, dim=1)
            return y
        else:
            # Train
            target = target.long().to(self.device)
            loss = self.cross_entropy(v_2, target)
            return loss