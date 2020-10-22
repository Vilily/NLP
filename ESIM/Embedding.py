""" Embed
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/22
"""
import torch
import torch.nn as nn
import torchtext.vocab as torch_Vocab
from vocab import Vocab

class EmbeddingGlove(nn.Module):
    def __init__(self, vocab, cache_dir='data/', dim=100, embedding_size=128 ,device=None, dropout=0.1):
        super(EmbeddingGlove, self).__init__()
        """
        :vocab :字典
        :cache_dir :存放Glove的路径
        """
        self.glove = torch_Vocab.GloVe(name='6B', dim=dim, cache=cache_dir)
        self.vocab = vocab
        self.stoi = {}
        # Model
        self.Embedding =nn.Parameter(torch.normal(mean=0, std=0.1, size=[11000, dim]), requires_grad=True)
        self.linear = nn.Linear(in_features=dim, out_features=embedding_size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X):
        """
        :X : (batch_size, length)
        :Y : (batch_size, length, embedding_size)
        """
        # 获取GloVe词向量
        Y = []
        for seq in X:
            sentence = []
            for index in seq:
                word = self.vocab.itos(index)
                # vec = self.glove.get_vecs_by_tokens(word)
                if word in self.glove.stoi:
                    vec = self.glove.get_vecs_by_tokens(word)
                else:
                    vec = self.Embedding[len(self.stoi)]
                    self.stoi[word] = len(self.stoi)
                sentence.append(vec)
            sentence = torch.stack(sentence, dim=0)
            Y.append(sentence)
        Y = torch.stack(Y, dim=0)
        # 映射到embedding
        output = self.linear(Y)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    vocab_path = 'data/snil_vocab.txt'
    vocab = Vocab(vocab_path)
    embed = EmbeddingGlove(vocab)
    a = [[23, 678, 9091], [2133, 1433, 0]]
    print(embed(a).shape)