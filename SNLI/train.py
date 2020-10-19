''' LSTM+Attention训练模型
@Author: Bao Wenjie
@Date: 2020/10/9
@Email: bwj_678@qq.com
'''
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from vocab import DataSet
import numpy as np
import torch
from vocab import Vocab
from LSTM import LSTMmodel



if __name__ == '__main__':
    train_data_path = 'data/snli_new_test.csv'
    dev_data_path = 'data/snli_new_dev.csv'
    test_data_path = 'data/snli_new_test.csv'
    vocab_path = 'data/snil_vocab.txt'

    BATCH_SIZE = 32
    max_length = 60
    embedding_size = 256
    hidden_size = 512
    lr = 0.001
    output_per_epochs = 100
    test_per_epochs = 300
    # 加载字典
    vocab = Vocab(vocab_path)
    # 创建数据集
    train_data_set = DataSet(train_data_path, vocab, max_length)
    test_data_set = DataSet(test_data_path, vocab, max_length)
    # 创建加载器
    train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_data_loader = DataLoader(test_data_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    # 是否用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # 模型初始化
    model = LSTMmodel(hidden_size=hidden_size,
                      embedding_size=embedding_size,
                      device=device,
                      calss_num=3,
                      is_word2word=False).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-3)
    # 开始训练
    for i in range(10):
        print('='*8 + '开始训练' + '='*8)
        model.train()
        loss_sum = 0
        for epoch, data in enumerate(train_data_loader):
            pre_X, pre_length, hyp_X, hyp_length, Y = data
            pre_X = pad_sequence(pre_X)
            hyp_X = pad_sequence(hyp_X)
            # tensor(batch_size, pre_length) 
            # tensor(batch_size, hyp_length) 
            # tensor(batch_size) 
            optimizer.zero_grad()
            loss = model(((pre_X, pre_length), (hyp_X, hyp_length)), Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            # 打印训练情况
            if((epoch + 1) % output_per_epochs == 0):
                print('itor: {}: epoch: {}  loss: {}'.format(i + 1, epoch + 1, loss_sum / output_per_epochs))
                loss_sum = 0
            ############################### 测试 ######################################
            if (epoch + 1) % test_per_epochs == 0:
                print('-'*8 + '开始测试' + '-'*8)
                with torch.no_grad():
                    accuracy = 0
                    model.eval()
                    for epoch, data in enumerate(test_data_loader):
                        pre_X, pre_length, hyp_X, hyp_length, target = data
                        pre_X = pad_sequence(pre_X)
                        hyp_X = pad_sequence(hyp_X)
                        target = target.long().to(device=device)
                        y = model(((pre_X, pre_length), (hyp_X, hyp_length))).detach()
                        accuracy += torch.sum(y == target).cpu()
                    print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, len(test_data_set), float(accuracy) / len(test_data_set)))
                    torch.save(model.state_dict(), 'output/lstm_model.pkl')
                model.train()

