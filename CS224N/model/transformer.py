import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, k, q, v, scale=None, attn_mask=None):
        ''' 前向传播
        @param q tensor(batch_size, seq_len, dim_per_head) : Question张量
        @param k tensor(batch_size, seq_len, dim_per_head) : 
        @param v tensor(batch_size, seq_len, dim_per_head) :
        @param scale tensor(ByteTensor)(batch_size, seq_len, seq_len) : 缩放因子
        @param attn_mask tensor(batch_size, seq_len, seq_len): masking张量

        @return context tensor(batch_size, seq_len, dim_per_head):
        @return attention tensor(batch_size, seq_len, seq_len)
        '''
        # Q*K
        attention = torch.bmm(q, k.transpose(1, 2)) #tensor (batch_size, seq_len, seq_len)
        # 缩放 1/sqrt(dk)
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 将mask为1的地方设为负无穷
            attention = attention.masked_fill_(attn_mask, -999999)
        # 计算softmax
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        # 乘V
        context = torch.bmm(attention, v) #tensor (batch_size, seq_len, dim_per_head)
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        ''' init
        @param model_dim (int):
        @param num_heads(int):
        @param dropout(int):
        '''
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        ''' 前向传播
        @param key tensor(batch_size, seq_len, model_dim):
        @param value tensor(batch_size, seq_len, model_dim):
        @param query tensor(batch_size, seq_len, model_dim):
        @param attn_mask tensor(batch_size, seq_len, seq_len):

        @return output tensor(batch_size, seq_len, model_dim)
        @return attention tensor(batch_size * num_heads, seq_len, seq_len)
        '''
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射
        key = self.linear_k(key) #(batch_size, seq_len, dim_per_head * num_heads)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 分割
        key = key.view(batch_size * num_heads, -1, dim_per_head) #(batch_size*num_heads, seq_len, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # masking
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1) #(batch_size*num_heads, seq_len, seq_len)
        
        # dot product attention
        scale = (key.size(-1) * num_heads) ** -0.5 # 1/sqrt(dim_per_head * num_heads)
        context, attention = self.dot_product_attention(key, query, value, scale, attn_mask)
        # context tensor(batch_size, * num_heads, seq_len, dim_per_heads)
        # attention tensor(batch_size * num_heads, seq_len, seq_len)

        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # context tensor(batch_size, seq_len, dim_per_head * num_heads)
        output = self.linear_final(context)
        # output tensor(batch_size, seq_len, model_dim)

        # dropout
        output = self.dropout(output)
        # 归一化和残差连接
        output = self.layer_norm(residual + output)
        return output, attention


class PositionEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        ''' init
        @param model_dim (int):模型维度
        @param max_seq_len (int):序列的最大长度
        '''
        super(PositionEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(1000, 2.0 * (j // 2) / model_dim) for j in range(model_dim)]
            for pos in range(max_seq_len)])
        # ndarray(max_seq_len, model_dim)
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, model_dim])
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)
        # tensor(1, model_dim)
        position_encoding = torch.cat((pad_row, position_encoding))
        # tensor(1 + max_seq_len, model_dim)

        self.position_encoding = nn.Embedding(max_seq_len + 1, model_dim)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        ''' 位置编码的前向传播
        @param input_len tensor(batch_size, 1):每个序列的长度
        '''
        max_len = max(input_len)
        tensor = torch.cuda.LongTensor# if input_len.is_cuda else torch.LongTensor

        input_pos = [list(range(1, len_ + 1)) + [0] * (max_len - len_) for len_ in input_len]
        input_pos = tensor(input_pos)
        return self.position_encoding(input_pos)

class FeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(FeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x):
        ''' feedforward的前向传播
        @param x tensor(batch_size, seq_len, model_dim):

        @return output tensor(batch_size, seq_len, model_dim)
        '''
        output = x.transpose(1, 2) #(batch_size, model_dim, seq_len)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2)) #(batch_size, model_dim, seq_len)
        # 归一化和残差连接
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    ''' Encoder的一层 '''
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        '''
        '''
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim=model_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.feed_forward = FeedForward(model_dim=model_dim,
                                        ffn_dim=ffn_dim,
                                        dropout=dropout)

    def forward(self, inputs, attn_mask=None):
        ''' EncoderLayer的前向传播
        @param inputs tensor(batch_size, seq_len, embed_size):输入的句子，已padding、embedding
        @param attn_mask tensor(batch_size, seq_len, seq_len):padding的遮罩

        @return output tensor(batch_size, seq_len, embed_size)
        @return attention tensor(batch_size * num_heads,seq_len, seq_len)
        '''
        # 多头自注意
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # context tensor(batch_size, seq_len, embed_size)
        # attention tensor(batch_size * num_heads, seq_len, seq_len)

        # feed forward
        output = self.feed_forward(context)
        # output tensor(batch_size, seq_len, embed_size)
        return output, attention

class Encoder(nn.Module):
    ''' Encoder实现 '''
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        '''init
        @param
        '''
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim=model_dim,
                          num_heads=num_heads,
                          ffn_dim=ffn_dim,
                          dropout=dropout)
                          for _ in range(num_layers)]
        )
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        ''' Encoder的前向传播
        @param inputs tensor(batch_size, seq_len):输入句子未embedding已padding
        @param inputs_len tensor(batch_size):输入句子的长度

        @return output tensor(batch_size, seq_len, embed_size):
        @return attention List[tensor(batch_size * num_heads, seq_len, seq_len)]:
        '''
        # 序列编码
        output = self.seq_embedding(inputs)
        # 位置编码
        output += self.pos_embedding(inputs_len)
        # output tensor(batch_size, seq_len, embed_size)
        # 生成padding遮罩
        self_attention_mask = padding_mask(inputs, inputs) #(batch_size, seq_len, seq_len)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            # output (batch_size, seq_len, embed_size)
            # attention (batch_size * num_heads, seq_len, seq_len)
            attentions.append(attention)
        return output, attentions

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim=model_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.feed_forward = FeedForward(model_dim=model_dim,
                                        ffn_dim=ffn_dim,
                                        dropout=dropout)
    def forward(self,
                dec_inputs,
                enc_outputs,
                self_attn_mask=None,
                context_attn_mask=None):
        ''' DecoderLayer前向传播
        @param dec_inputs tensor(batch_size, tgt_seq_len, model_dim): 目标语句的输入已padding已embeddig
        @param enc_output tensor(batch_size, src_seq_len, model_dim): encoder的output
        @param self_attn_mask tensor(batch_size, tgt_seq_len, tgt_seq_len):自注意力的padding遮罩
        @param context_attn_mask tensor(batch_size, src_seq_len, tgt_seq_len):encoder-decoder attention的上下文遮罩

        @return dec_output tensor(batch_size, tgt_seq_len, model_dim):
        @return self_attention tensor(batch_size, tgt_seq_leb, tgt_seq_len)
        @return context_attention tensor(batch_size, tgt_seq_len, src_seq_len)
        '''
        # 多头自注意力计算
        dec_output, self_attention = self.attention(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask
        )
        # dec_output tensor(batch_size, seq_len, model_dim)
        # context_attention tensor(batch_size * num_heads, seq_len, seq_len)

        # encoder-decoder注意力计算
        dec_output, context_attention = self.attention(
        enc_outputs, enc_outputs, dec_output, context_attn_mask)
        # dec_output tensor(batch_size, tgt_seq_len, model_dim)
        # context_attention tensor(batch_size * num_heads, tgt_seq_len, src_seq_len)

        dec_output = self.feed_forward(dec_output) # (batch_size, tgt_seq_len, model_dim)
        return dec_output, self_attention, context_attention

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim=model_dim,
                          num_heads=num_heads,
                          ffn_dim=ffn_dim,
                          dropout=dropout)
                        for _ in range(num_layers)]
        )

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        ''' Decoder前向传播
        @param inputs tensor(batch_size, seq_len):输入语句未embedding已padding
        @param inputs_len tensor(batch_size):目标句子的长度
        @param enc_ioutput tensor(batch_size, seq_len, model_dim):Encoder的输出output
        @context_attn_mask tensor(batch_size, src_seq_len, tgt_seq_len):encoder-decoder attention

        @return output tensor(batch_size, tgt_seq_len, model_dim)
        @return self_attentions List[tensor(batch_size, tgt_seq_len, tgt_seq_len)]
        @return context_attentions List[tensor(batch_size, tgt_seq_len, src_seq_len)]
        '''
        # 词嵌入
        output = self.seq_embedding(inputs)
        # 位置嵌入
        output += self.pos_embedding(inputs_len)
        # output tensor(batch_size, seq_len, model_dim)

        # padding遮罩
        self_attention_padding_mask = padding_mask(inputs, inputs) #(batch_size, seq_len, seq_len)
        # 上下文遮罩
        seq_mask = sequence_mask(inputs).cuda() #(batch_size, seq_len, seq_len)
        # 合并遮罩
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0).cuda() #(batch_size, seq_len, seq_len)
        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask
            )
            # output tensor(batch_size, tgt_seq_len, model_dim)
            # self_attn tensor(batch_size, tgt_seq_len, tgt_seq_len)
            # context_attn tensor(batch_size, tgt_seq_len, src_seq_len)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        
        return output, self_attentions, context_attentions


class Transformer(nn.Module):
    def __init__(self,
                 device,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        ''' tansformer模型初始化
        @param src_vacab_size (int):输出字典大小
        @param src_max_len (int):输入的句子的最大长度
        @param tgt_vocab_size (int):输出的字典大小
        @param tgt_max_len (int):输出的句子最大长度
        @param num_layers (int):堆叠的层数
        @param ffn_dim (int):feed forward层的维度
        @param dropout (float):dropout
        '''
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(vocab_size=src_vocab_size,
                               max_seq_len=src_max_len,
                               num_layers=num_layers,
                               model_dim=model_dim,
                               num_heads=num_heads,
                               ffn_dim=ffn_dim,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=tgt_vocab_size,
                               max_seq_len=tgt_max_len,
                               num_layers=num_layers,
                               model_dim=model_dim,
                               num_heads=num_heads,
                               ffn_dim=ffn_dim,
                               dropout=dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        ''' transformer前向传播
        @param src_seq tensor(batch_size, seq_len):未embedding已padding的输入句子
        @param src_len tensor(batch_size):输入句子每句话的长度
        @param tgt_seq tensor(batch_size, seq_len):未embedding已padding的输出句子
        @param tgt_len tensor(batch_size):输出句子的每句话长度
        
        @return output tensor()
        '''
        # 将list转成tensor(未embed)
        src_seq = torch.tensor(src_seq, dtype=torch.int64, device=self.device)
        # self.vocab.src.to_input_tensor(pad_sources_batch, device=self.device)   # Tensor: (batch_size, seq_len)
        # src_len = torch.tensor(src_len, dtype=torch.int64, device=self.device, requires_grad=False)
        tgt_seq = torch.tensor(tgt_seq, dtype=torch.int64, device=self.device)
        # tgt_len = torch.tensor(tgt_len, dtype=torch.int16, device=self.device)

        # 生成padding的遮罩
        context_attn_mask = padding_mask(tgt_seq, src_seq) #(batch_size, src_seq_len, tgt_seq_len)

        output, enc_self_attn = self.encoder(src_seq, src_len)
        # output tensor(batch_size, seq_len, embed_size)
        # enc_self_attn List[tensor(batch_size * num_heads, seq_len, seq_len)]
        output, dec_self_attn, context_attn = self.decoder(
            tgt_seq, tgt_len, output, context_attn_mask
        )
        # output (batch_size, tgt_seq_len, model_dim)
        # dec_self_attn list[(batch_size, tgt_seq_len, tgt_seq_len)]
        # context_attn list[(batch_size, tgt_seq_len, src_seq_len)]
        output = self.linear(output)
        # (batch_size, tgt_seq_len, vocab_size)
        output = self.softmax(output)
        # 计算loss
        # tgt_seq = tgt_seq[:, 1:]
        # target_padded: tensor (batch, seq_len, vocab_len)
        target_masks = (tgt_seq != 0).float()
        # print(tgt_seq)
        # print(target_masks)
        # target_masks: tensor (batch_size, seq_len)
        target_gold_words_log_prob = torch.gather(output, index=tgt_seq.unsqueeze(-1), dim=-1).squeeze(-1)
        target_gold_words_log_prob = target_gold_words_log_prob * target_masks
        # target_gold_words_log_prob: tensor (batch_size, seq_len)
        scores = target_gold_words_log_prob.sum(dim=1)
        # return output, enc_self_attn, dec_self_attn, context_attn
        return scores


def padding_mask(seq_k, seq_q):
    ''' 给padding(0)好的序列生成遮罩
    @param seq_k tensor(batch_size, K_seq_len): 目标序列
    @param seq_q tensor(batch_size, Q_seq_len):
    
    @return pad_mask tensor(batch_size, Q_seq_len, K_seq_len)
    '''
    len_q = seq_q.size(1) # seq_len
    pad_mask = seq_k.eq(0) #tensor(batch_size, seq_len)
    pad_mask = pad_mask.unsqueeze(2).expand(-1, -1, len_q) # tensor (batch_size, K_seq_len, Q_seq_len)
    return pad_mask

def sequence_mask(seq):
    ''' 生成对句子上下文的mask
    @seq tensor(batch_size, seq_len)
    @mask tensor(batch_size, seq_len, seq_len): 全为1的上右三角矩阵，不包括对角线
    '''
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8).to(dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1) #[bathc_size, seq_len, seq_len]
    return mask

if __name__ == "__main__":
    # model = Transformer(device=torch.device('cpu'),
    #                         src_vocab_size = 100,
    #                         src_max_len = 200,
    #                         tgt_vocab_size = 100,
    #                         tgt_max_len = 100,
    #                         num_layers=3,
    #                         model_dim=512,
    #                         num_heads=8,
    #                         ffn_dim=2048,
    #                         dropout=0.2)
    pass