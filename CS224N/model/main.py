# -*- coding: utf-8 -*-
import argparse
from sklearn.model_selection import train_test_split
from data_helper import DATA_PATH, MODEL_PATH
import pandas as pd
from modelNet import *
from data_helper import *
from vocab import Vocab, VocabEntry

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()


class Main(object):
    def __init__(self):
        self.epochs = args.EPOCHS
        self.BATCH = args.BATCH
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()):
            self.device = torch.device('cuda:0')

    def deal_with_data(self):
        # 加载词典
        # src_vocab = VocabEntry(os.path.join(DATA_PATH, 'MedicalQA/ask_fr.dict'))
        # tgt_vocab = VocabEntry(os.path.join(DATA_PATH, 'MedicalQA/ans_fr.dict'))
        src_vocab = VocabEntry(os.path.join(DATA_PATH, 'ask_fr.dict'), isAsk=True)
        tgt_vocab = VocabEntry(os.path.join(DATA_PATH, 'ans.json'))
        self.vocab = Vocab(src_vocab, tgt_vocab)
        print(len(src_vocab), len(tgt_vocab))
        # 加载数据集
        # self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalQA/train.csv'))
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
        # 划分训练集、测试集
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.01, random_state=678, shuffle=True)
        # 计算每个epoch的batch数
        self.steps_per_epoch = int(len(self.train_data.index) / self.BATCH)
        # 将文本变成id
        self.source_train, self.target_train = read_data_id(self.train_data, self.vocab.src, self.vocab.tgt)
        self.source_test, self.target_test = read_data_id(self.test_data, self.vocab.src, self.vocab.tgt)
        print('=='*8+'数据处理完成！'+'=='*8)

    def get_train_batch_data(self):
        ''' 返回一个batch的训练数据
        @return pad_target_batch: List[List[int]], 一个batch的padding好的目标数据, [batch_size, seq_len]
        @return pad_sources_batch: List[List[int]], 一个batch的padding好的输入数据, [batch_size, seq_len]
        @return target_len_batch, source_len_batch: List[int], 句子原始长度
        '''
        return get_batches(self.target_train, self.source_train, self.BATCH)

    def train(self):
        decoder_vocab_size = len(self.vocab.tgt)
        # Embedding Size
        embedding_size = 256
        # Learning Rate
        learning_rate = 0.003
        # hidden size
        hidden_size = 256
        # clip grad
        clip_grad = 100
        # log_every
        log_every = 10

        # 构造graph
        self.model = ChatBotModel(self.BATCH, embedding_size, hidden_size, vocab=self.vocab, device=self.device, droprate=0.2)
        #self.load_model(self.model)
        self.model = self.model.to(device=self.device)
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        print("--"*8+'开始训练'+'--'*8)
        for epoch in range(self.epochs):
            self.model.train()
            for train_iter, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(get_batches(
                            self.target_train, self.source_train, self.BATCH, self.vocab.src,
                            self.vocab.tgt)):
                batch_size = len(targets_batch)
                # forward pass
                example_losses = -self.model(targets_batch, sources_batch, sources_lengths)
                # compute loss
                batch_loss = example_losses.sum()
                loss = batch_loss / batch_size
                loss_val = loss.item()
                optimizer.zero_grad()
                # backward pass
                loss.backward(retain_graph=False)
                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                # optimize
                optimizer.step()
                torch.cuda.empty_cache()
                # if train_iter % log_every == 0:
                print('Epoch: {} | Train_iter: {} | Train Loss: {}'.format(epoch, train_iter, loss_val))
            # 在测试集测试
            print("--"*8+'开始测试'+'--'*8)
            self.model.eval()
            with torch.no_grad():
                loss_ = 0
                for test_iter, (targets_batch_test, sources_batch_test, targets_lengths, sources_lengths_test) in enumerate(get_batches(
                    self.target_test, self.source_test, args.BATCH, self.vocab.src,
                    self.vocab.tgt)):

                    batch_size = len(targets_batch)
                    example_losses = -self.model(targets_batch, sources_batch, sources_lengths) # (batch_size,)
                    batch_loss = example_losses.sum()
                    loss = batch_loss / batch_size
                    loss_ += loss.item()
                print('test-loss:{}'.format(loss_))
            self.save_model(self.model)

    def save_model(self, model):
        torch.save(model.state_dict(), 'data/output/model/model.pkl')
    
    def load_model(self, model):
        model.load_state_dict(torch.load('data/output/model/model.pkl'))


if __name__ == '__main__':
    main = Main()
    # # main.download_data()
    main.deal_with_data()
    main.train()
    exit(0)
