import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, hidden_size, embedding_size, device, calss_num, batch_size=1, is_word2word=False):
        """ 初始化
        :param hidden_size: lstm隐藏层大小
        :param embedding_size: 词嵌入向量大小
        :param device: 训练设备
        :param class_num: 分类数量
        """
        super(LSTMmodel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.is_word2word = is_word2word
        self.lstmA = nn.LSTMCell(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 bias=True).to(device)
        self.lstmB = nn.LSTMCell(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 bias=True).to(device)
        self.embedding = nn.Embedding(num_embeddings=40000, embedding_dim=embedding_size).to(device)
        self.h_0 = torch.zeros((batch_size, hidden_size), requires_grad=False, device=device)
        self.c_0 = torch.zeros((batch_size, hidden_size), requires_grad=False, device=device)
        # Attention
        self.W_y = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
        self.W_h = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
        # word2word Attention
        if is_word2word:
            self.W_r = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
            self.W_t = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
        self.w = torch.randn(size=(batch_size, 1, hidden_size), requires_grad=True, device=device)
        self.tanh = nn.Tanh()
        self.attn_softmax = nn.Softmax(dim=1)
        # final sentence-pair representation
        self.W_p = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
        self.W_x = torch.randn(size=(batch_size, hidden_size, hidden_size), requires_grad=True, device=device)
        # calssification
        self.classify = nn.Linear(in_features=hidden_size, out_features=calss_num, bias=True).to(device)
        self.softmax = nn.Softmax(dim=0)
        self.crossLoss = nn.CrossEntropyLoss()

    def forward(self, X, target=None):
        """ 前向传播
        :param X(premise_size, hypothesis_size)
        :param target int
        """
        X_pre, X_hyp = X
        X_pre = X_pre.long().to(device=self.device) #(batch_size, pre_length)
        X_hyp = X_hyp.long().to(device=self.device) #(batch_size, hyp_length)
        # 词嵌入
        X_pre_embed = self.embedding(X_pre) #(batch_size, premise_length, embedding_size)
        X_hyp_embed = self.embedding(X_hyp) #(batch_size, hypothesis_length, embedding_size)
        X_pre_embed = X_pre_embed.permute([1, 0, 2]) # (premise_length, batch_size, embedding_size)
        X_hyp_embed = X_hyp_embed.permute([1, 0, 2]) # (hypothesis_length, batch_size, embedding_size)
        h_pre_list = [] # (pre_length, batch_size, hidden_size)
        c_pre_list = [] # (pre_length, batch_size, hidden_size)
        h_t  =self.h_0
        c_t = self.c_0
        # premise LSTM
        for i in range(X_pre.shape[1]):
            x_t = X_pre_embed[i]                     # (batch_size, embedding_size)
            (h_t, c_t) = self.lstmA(x_t, (h_t, c_t)) # (batch_size, hidde_size)
            c_pre_list.append(c_t)
            h_pre_list.append(h_t)
        # hypthsis LSTM
        c_hyp_list = [] #(hyp_length, batch_size, hidden_size)
        h_hyp_list = [] #(hyp_length, batch_size, hidden_size)
        for i in range(X_hyp.shape[1]):
            x_t = X_hyp_embed[i]
            (h_t, c_t) = self.lstmB(x_t, (h_t, c_t))
            c_hyp_list.append(c_t)
            h_hyp_list.append(h_t)
        # Attention
        pre = (h_pre_list, c_pre_list)
        hyp = (h_hyp_list, c_hyp_list)
        if self.is_word2word:
            # word2word
            r = self.word2wordAttention(pre, hyp)
        else:
            # normal
            r = self.attention(pre, hyp) #(batch_size, hidden_size)
        # final sentence-pair representation
        h_n = h_hyp_list[-1].unsqueeze(dim=2) #(batch_size, hidden_size, 1)
        r_ = torch.bmm(self.W_p, r) #(batch_size, hidden_size, 1)
        h_ = torch.bmm(self.W_x, h_n) #(batch_size, hidden_size, 1)
        h_star = self.tanh(r_ + h_).squeeze() #(batch_size, hidden_size)
        possibility = self.classify(h_star) #(batch_size, class_num)
        # classfication
        if target is None:
            # predict
            predict = torch.argmax(self.softmax(possibility), dim=1)
            return predict
        else:
            # tarin
            target = target.long().to(device=self.device)
            loss = self.crossLoss(possibility, target)
            return loss

    
    def attention(self, pre, hyp):
        """ attention
        :param pre (h, c): premise的LSTM结果
        :param hyp (h, c): hypothesis的LSTM结果
        :return r:(batch_size, hidden_size)
        """
        h_pre_list, c_pre_list = pre # (pre_length, batch_size, hidden_size)
        h_hyp_list, c_hyp_list = hyp # (hyp_length, batch_size, hidden_size)
        Y = torch.stack(h_pre_list)  # (pre_length, batch_size, hidden_size)
        Y = Y.permute([1, 2, 0])     # (batch_size, hidden_size, pre_length)
        h_n = h_hyp_list[-1].unsqueeze(dim=2)      #(batch_size, hidden_size, 1)
        M = torch.bmm(self.W_y, Y) + torch.bmm(self.W_h, h_n) #(batch_size, hidden_size, pre_length)
        M = self.tanh(M)
        a = torch.bmm(self.w, M) # (bacth_size, 1, pre_length)
        alpha = self.attn_softmax(a).permute([0, 2, 1]) # (bacth_size, pre_length, 1)
        r = torch.bmm(Y, alpha) # (batch_size, hidden_size, 1)
        return r

    def word2wordAttention(self, pre, hyp):
        """ 单词水平的attention
        :param pre (pre_h, pre_c): premise的LSTM结果
        :param hyp (hyp_h, hyp_c): hypothesis的LSTM结果
        """
        h_pre_list, c_pre_list = pre # (pre_length, batch_size, hidden_size)
        h_hyp_list, c_hyp_list = hyp # (hyp_length, batch_size, hidden_size)
        Y = torch.stack(h_pre_list)  # (pre_length, batch_size, hidden_size)
        Y = Y.permute([1, 2, 0])     # (batch_size, hidden_size, pre_length)
        r_t = torch.zeros([self.batch_size, self.hidden_size, 1], device=self.device)
        for h_t in h_hyp_list:
            h_t = h_t.unsqueeze(dim=2)
            M = torch.bmm(self.W_y, Y) + torch.bmm(self.W_h, h_t) + torch.bmm(self.W_r, r_t) #(batch_size, hidden_size, pre_length)
            M = self.tanh(M)
            a = torch.bmm(self.w, M) # (bacth_size, 1, pre_length)
            alpha = self.attn_softmax(a).permute([0, 2, 1]) # (bacth_size, pre_length, 1)
            r_t = torch.bmm(Y, alpha) + self.tanh(torch.bmm(self.W_t, r_t)) # (batch_size, hidden_size, 1)
        return r_t
        

