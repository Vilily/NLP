""" End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/22
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, device, embedding_size, p=0.1, n=5):
        super(CNN, self).__init__()
        """ CNN初始化
        """
        # dropout
        self.dropout = nn.Dropout(p=p)
        # convolution
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(n, 1), stride=1, padding=(2, 0))
        # max pooling
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, embedded_char):
        """ 卷积
        :embedded_char (batch_size, sent_length, word_length, embedding_size)
        :char_representation (batch_size, sent_length, hidden_size)
        """
        char_representation = []
        for sent_embedded_char in embedded_char:
            sent = []
            for word_embedded_char in sent_embedded_char: 
                # (word_length, embedding_size)
                word_embedded_char = word_embedded_char.unsqueeze(dim=0).unsqueeze(dim=0)
                # (1, 1, word_length, embedding_size)
                word_embedded_char = self.dropout(word_embedded_char) 
                word_embedded_char = self.conv(word_embedded_char).squeeze(dim=0) # (1, word_length + pad - k + 1, hidden_size)
                word_embedded_char = word_embedded_char.permute([0, 2, 1])
                # (1, hidden_size, word_length + pad - k + 1)
                word_embedded_char = self.max_pooling(word_embedded_char).squeeze(dim=2) 
                # (1, hidden_size) 
                sent.append(word_embedded_char)
            sent = torch.cat(sent, dim=0)
            char_representation.append(sent) # (sent_length, hidden_size)
        char_representation = torch.stack(char_representation, dim=0)
        return char_representation


if __name__ == "__main__":
    cnn = CNN(torch.device('cpu'), embedding_size=5)
    embedded_char = torch.randn(size=[3, 4, 6, 5])
    print(cnn(embedded_char).shape)