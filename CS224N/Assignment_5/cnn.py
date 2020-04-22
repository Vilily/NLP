#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, char_dim:int, embed_char=50, k=5):

        '''
        @param char_dim: int of the dimention of embedded word

        @param filters: int of the filters of output

        @param k: int of kernel size
        '''
        super(CNN, self).__init__()
        self.max_word_length = 21
        self.embed_char = embed_char
        self.char_dim = char_dim
        self.convd = nn.Conv1d(in_channels=embed_char, out_channels=char_dim, kernel_size=k, padding=(k-1)//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        @param x: Tensor of input data of shape (sentence_length, batch_size, max_word_length, embed_char)

        @param y: Tensor of output data of shape (batch_size, char_dim)
        '''
        x = x.permute(0, 1, 3, 2)#(sentence_length, batch_size, embed_char, max_word_length)
        self.batch_size = x.size()[1]
        self.sentence_length = x.size()[0]
        x = x.reshape(-1,  self.embed_char, self.max_word_length)
        x = self.convd(x)   #(batch_size, char_dim, sentence_length)
        x = x.reshape(self.sentence_length, self.batch_size, self.char_dim, self.max_word_length)
        #(sentence_length, batch_size, char_dim, max_word_length)
        x = self.relu(x)    #()
        y = torch.max(x, dim=3)    #(batch_size, char_dim)
        y = y.values
        return y

### END YOUR CODE

