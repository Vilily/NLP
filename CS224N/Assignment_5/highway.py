#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch

class Highway(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout:float):
        '''

        @param input_size: int of word vector dimention
        @param output_siez: 
        '''
        super(Highway, self).__init__()
        self.l1 = nn.Linear(input_size, output_size, bias=True)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(input_size, output_size, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        @param x: Tensor of shape (sentence_length, batch_size, char_dim).
        @param Xembed: Tensor of shape (sentence_length, batch_size, char_dim).
        '''
        Xproj = self.relu(self.l1(x))
        Xgate = self.softmax(self.l2(x))
        Xhighway = torch.mul(Xproj, Xgate) + torch.mul((1 - Xgate), x)
        Xembed = self.dropout(Xhighway)
        return Xembed


### END YOUR CODE 

