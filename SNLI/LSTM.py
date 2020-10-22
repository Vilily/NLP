import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMmodel(nn.Module):
    def __init__(self, hidden_size, embedding_size, device, calss_num, is_word2word=False):
        """ 初始化
        :param hidden_size: lstm隐藏层大小
        :param embedding_size: 词嵌入向量大小
        :param device: 训练设备
        :param class_num: 分类数量
        """
        super(LSTMmodel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.is_word2word = is_word2word
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=40000, embedding_dim=embedding_size).to(device)
        # LSTM
        self.lstmA = nn.LSTM(input_size=embedding_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bias=True,
                             batch_first=True)
        self.lstmB = nn.LSTM(input_size=embedding_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bias=True,
                             batch_first=True)
        # transformation of the states
        u_min = -0.5
        u_max = 0.5
        # Attention
        self.W_y = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # word2word Attention
        if is_word2word:
            self.W_r = nn.Parameter(torch.randn(1, hidden_size, hidden_size))
            self.W_t = nn.Parameter(torch.randn(1, hidden_size, hidden_size))
        self.w = nn.Parameter(torch.randn(1, hidden_size))
        self.tanh = nn.Tanh()
        # final sentence-pair representation
        self.W_p = nn.Parameter(torch.randn(1, hidden_size, hidden_size))
        self.W_x = nn.Parameter(torch.randn(1, hidden_size, hidden_size))
        # calssification
        self.classify = nn.Linear(in_features=hidden_size, out_features=calss_num, bias=True).to(device)
        self.softmax = nn.Softmax(dim=1)
        self.crossLoss = nn.CrossEntropyLoss()

    def forward(self, X, hyp2pre=None, target=None):
        """ 前向传播
        :param X(premise_size, hypothesis_size)
        :param target int
        """
        pre_data, hyp_data = X
        pre_x, pre_length = pre_data
        hyp_x, hyp_length = hyp_data
        pre_x = pre_x.long().to(device=self.device) # (batch_size, pre_length)
        hyp_x = hyp_x.long().to(device=self.device) # (batch_size, hyp_length)
        # Embedding
        X_pre_embed = self.embedding(pre_x) # (batch_size, pre_length, embedding_size)
        X_hyp_embed = self.embedding(hyp_x) # (batch_size, hyp_length, embedding_size)
        # Pack
        X_pre_embed = pack_padded_sequence(X_pre_embed, pre_length, batch_first=True)
        X_hyp_embed = pack_padded_sequence(X_hyp_embed, hyp_length, batch_first=True)
        # Premise LSTM
        X_pre_hidden, hidden_state = self.lstmA(X_pre_embed) # (batch_size, pre_length, hidden_size); (batch_size, 2*hidden_size, hidden_size)
        X_pre_hidden, X_pre_hidden_length = pad_packed_sequence(X_pre_hidden, batch_first=True)

        # Hypothesis LSTM
        X_hyp_hidden, _ = self.lstmB(X_hyp_embed, hidden_state) # (batch_size, hyp_length, hidden_size)
        X_hyp_hidden, X_hyp_hidden_length = pad_packed_sequence(X_hyp_hidden, batch_first=True)
        X_hyp_hidden = X_hyp_hidden[hyp2pre,:,:]
        # Attention
        if self.is_word2word:
            # word2word
            r = self.word2wordAttention((X_pre_hidden, X_pre_hidden_length), (X_hyp_hidden, X_hyp_hidden_length))
        else:
            # normal
            r = self.attention((X_pre_hidden, X_pre_hidden_length), (X_hyp_hidden, X_hyp_hidden_length)) # (batch_size, hidden_size, 1)

        # Final sentence-pair representation
        h_n = torch.stack([_[X_hyp_hidden_length[i] - 1] for i, _ in enumerate(X_hyp_hidden)], dim=0).unsqueeze(dim=2)
        r_ = torch.matmul(self.W_p, r) #(batch_size, hidden_size, 1)
        h_ = torch.matmul(self.W_x, h_n) #(batch_size, hidden_size, 1)
        h_star = self.tanh(r_ + h_).squeeze() #(batch_size, hidden_size)
        possibility = self.classify(h_star) #(batch_size, class_num)

        # Classfication
        if target is None:
            # predict
            predict = torch.argmax(self.softmax(possibility), dim=1)
            return predict
        else:
            # tarin
            target = target.long().to(device=self.device)
            loss = self.crossLoss(possibility, target)
            return loss

    
    def attention(self, pre: torch, hyp: torch):
        """ attention
        :param pre (h, c), pre_length (batch_size): premise的LSTM结果 (batch_size, pre_length, hidden_size)
        :param hyp (h, c), hyp_length (batch_size): hypothesis的LSTM结果 (batch_size, hyp_length, hidden_size)
        :return r:(batch_size, hidden_size, 1)
        """
        # Unpack
        pre, pre_length = pre # (batch_size, pre_length, hidden_size)
        hyp, hyp_length = hyp # (batch_size, hyp_length, hidden_size)
        r_list = []
        ############################ Single ################################
        for i, h_n, in enumerate(hyp):
            h_n = h_n[hyp_length[i] - 1].unsqueeze(dim=1) # (hidden_size, 1)
            Y = pre[i, :pre_length[i]] # (pack_pre_length, hidden_size)
            Y = Y.permute([1, 0])      # (hidden_size, pack_pre_length)
            M = torch.mm(self.W_y, Y) + torch.mm(self.W_h, h_n) # (hidden_size, pack_pre_length)
            M = self.tanh(M)
            a = torch.mm(self.w, M) # (1, pack_pre_length)
            alpha = self.softmax(a).permute([1, 0]) # (pakc_pre_length, 1)
            r = torch.mm(Y, alpha) # (hidden_size, 1)
            r_list.append(r)
        r_list = torch.stack(r_list, dim=0) # (batch_size, hidden_size, 1)
        return r_list

        ################################# batch ###################################
        # h_n = torch.stack([hyp_[hyp_length[i] - 1] for i, hyp_ in enumerate(hyp)], dim=0).unsqueeze(dim=2) 
        # # (batch_size, hidden_size, 1)
        # h_n = hyp[:,hyp_length - 1,:].unsqueeze(dim=2)      # (batch_size, hidden_size, 1)
        # M = torch.matmul(self.W_y, Y)
        # M += torch.matmul(self.W_h, h_n) # (batch_size, hidden_size, pre_length)
        # M = self.tanh(M)
        # a = torch.matmul(self.w, M).squeeze(dim=1)  # (bacth_size, 1, pre_length)
        # alpha = self.softmax(a).unsqueeze(dim=2)    # (bacth_size, pre_length, 1)
        # r = torch.matmul(Y, alpha)                  # (batch_size, hidden_size, 1)
        # return r

    def word2wordAttention(self, pre, hyp):
        """ 单词水平的attention
        :param pre (h, c), pre_length (batch_size): premise的LSTM结果 (batch_size, pre_length, hidden_size)
        :param hyp (h, c), hyp_length (batch_size): hypothesis的LSTM结果 (batch_size, hyp_length, hidden_size)
        :return r:(batch_size, hidden_size)
        """
        # Unpack
        pre, pre_length = pre # (batch_size, pre_length, hidden_size)
        hyp, hyp_length = hyp # (batch_size, hyp_length, hidden_size)
        r_list = []
        ################################# batch ###########################
        Y = pre.permute([0, 2, 1])  # (batch_size, hidden_size, pre_length)
        r_t = torch.randn([self.batch_size, self.hidden_size, 1], device=self.device)
        for h_t in hyp:
            h_t = h_t.unsqueeze(dim=2)
            M = torch.bmm(self.W_y, Y) + torch.matmul(self.W_h, h_t) + torch.matmul(self.W_r, r_t) #(batch_size, hidden_size, pre_length)
            M = self.tanh(M)
            a = torch.matmul(self.w, M) # (bacth_size, 1, pre_length)
            alpha = self.attn_softmax(a).permute([0, 2, 1]) # (bacth_size, pre_length, 1)
            r_t = torch.matmul(Y, alpha) + self.tanh(torch.bmm(self.W_t, r_t)) # (batch_size, hidden_size, 1)
        return r_t
        

