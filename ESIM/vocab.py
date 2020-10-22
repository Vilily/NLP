import pandas as pd
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np
import torchtext.vocab as vocab

def generate_vocab(path, vocab_save_path):
    """ 生成字典
    """
    vocab = set()
    for i, path_ in enumerate(path):
        # 读取文件
        data = pd.read_csv(path_)
        for index, itor in data.iterrows():
            s = itor['sentence1']
            if not isinstance(s, str):
                s = '_pad_'
            words = s.split()
            for word in words:
                stoi.add(word)
            s = itor['sentence2']
            if not isinstance(s, str):
                s = '_pad_'
            words = s.split()
            for word in words:
                stoi.add(word)
    # 保存单词表
    stoi.discard('_pad_')
    with open(vocab_save_path, mode='w+', encoding='ascii') as file:
        for word in stoi:
            file.write(word + '\n')

def clear_data(path, save_path):
    """ 删除多余列/小写转大写/strip/删除末尾标点
    """
    data = pd.read_csv(path, delimiter='\t')
    # 清洗数据
    data_X = data[['gold_label', 'sentence1', 'sentence2']].values.tolist()
    for index, row in enumerate(data_X):
        if index % 100 == 0:
            print(index)
        sent1 = row[1]
        if not isinstance(sent1, str):
            sent1 = '_pad_'
        sent1 = sent1.lower()
        sent1 = sent1.replace(',', ' , ').replace('.', ' . ').replace('\"', ' " ').replace(';', ' ; ').replace('!', ' ! ')

        sent2 = row[2]
        if not isinstance(sent2, str):
            sent2 = '_pad_'
        sent2 = sent2.lower()
        sent2 = sent2.replace(',', ' , ').replace('.', ' . ').replace('"', ' " ').replace(';', ' ; ').replace('!', ' ! ')
        data_X[index] = [row[0], sent1, sent2]
    data = pd.DataFrame(data_X, columns=['gold_label', 'sentence1', 'sentence2'])
    data.to_csv(save_path, index=False)

class Vocab():
    def __init__(self, vocab_path):
        super().__init__()
        self._stoi = {}
        self._itos = {}
        self._pad_ = 0
        self._stoi['_pad_'] = self._pad_
        self._itos[self._pad_] = '_pad_'
        self.load_file(path=vocab_path)
        self.max_l = 0

    
    def load_file(self, path):
        """ 加载字典文件
        """
        with open(path, mode='r') as file:
            data = file.readlines()
        begin = 2
        for word in data:
            word = word.strip()
            if word not in self._stoi:
                self._stoi[word] = begin
                self._itos[begin] = word
                begin += 1
        return self._stoi
    
    def __len__(self):
        return len(self._stoi)
    
    def __getitem__(self, word):
        """
        根据word查询index
        """
        return self._stoi.get(word, self._pad_)
    
    def itos(self, index):
        """
        根据index查询word
        """
        return self._itos.get(index, None)
    
    def sents2indexs(self, sents, max_length):
        """ 句子转index
        :param sents ([str])
        :return indexs ndarray(n, max_length)
        """
        indexs = []
        lengths = []
        for sent in sents:
            index, length = self.sent2index(sent, max_length)
            indexs.append(index)
            lengths.append(length)
        return (indexs, lengths)
    
    def sent2index(self, sent:str, max_length: None):
        """ sent(str) 转 ndarray(max_length)
        """
        sent = sent.split()
        length = len(sent)
        if max_length is None:
            # 不padding
            sent = [self.__getitem__(word) for word in sent]
        else:
            # padding
            if len(sent) > max_length:
                sent = [self.__getitem__(word) for word in sent[:max_length]]
                length = max_length
            else:
                sent = [self.__getitem__(word) for word in sent] + [self._pad_]* (max_length - len(sent))
        return (sent, length)


class DataSet(Dataset):
    def __init__(self, path, vocab: Vocab, max_length):
        super(DataSet, self).__init__()
        self.vocab = vocab
        self.data_Y = None
        self.data_X_1 = None
        self.data_X_2 = None
        self.length_X_1 = None
        self.length_X_2 = None
        self.max_length = max_length
        self.load_data(path)

    def __getitem__(self, index):
        return (self.data_X_1[index], self.length_X_1[index], self.data_X_2[index], self.length_X_2[index], self.data_Y[index])

    def __len__(self):
        return len(self.data_Y)

    def load_data(self, path):
        ''' 加载数据
        :param path(str): 文件路径
        :param vocab(Vocab): 字典
        :return data ([sent1, sent2, label]):
        '''
        # 读取数据
        data = pd.read_csv(path)
        # label映射到0/1/2
        labelMap = {'neutral':0, 'entailment':1, 'contradiction':2}
        data_Y = data['gold_label'].values
        self.data_Y = [labelMap.get(label, 0) for label in data_Y]
        # 获取X
        data_X_1 = data['sentence1'].values
        data_X_2 = data['sentence2'].values
        # word映射到int
        self.data_X_1, self.length_X_1 = self.vocab.sents2indexs(data_X_1, self.max_length)
        self.data_X_2, self.length_X_2 = self.vocab.sents2indexs(data_X_2, self.max_length)
        return (self.data_X_1, self.length_X_1, self.data_X_2, self.length_X_2 ,self.data_Y)
        
def collate_func(X):
    """
    :pre_X (tensor):从大到小排列的pre
    :pre_length (tensor): pre长度
    :hyp_X (tensor): 从大到小排列的hyp
    :hyp_length (tensor): hyp长度
    :Y (tensor):target和pre对应
    :pre_indices (tensor): hyp to pre
    """
    pre_X = []
    pre_length = []
    hyp_X = []
    hyp_length = []
    Y = []
    for i in X:
        pre_X.append(i[0])
        pre_length.append(i[1])
        hyp_X.append(i[2])
        hyp_length.append(i[3])
        Y.append(i[4])
    pre_length = torch.tensor(pre_length)
    pre_X = torch.tensor(pre_X)
    hyp_X = torch.tensor(hyp_X)
    hyp_length = torch.tensor(hyp_length)
    Y = torch.tensor(Y)
    # Hypothsis Sort
    hyp_length, hyp_indices = torch.sort(hyp_length, descending=True)
    hyp_X = hyp_X[hyp_indices]
    pre_X = pre_X[hyp_indices]
    pre_length = pre_length[hyp_indices]
    Y = Y[hyp_indices]
    # Premise Sort
    pre_length, pre_indices = torch.sort(pre_length, descending=True)
    pre_X = pre_X[pre_indices]
    Y = Y[pre_indices]
    # Pack
    pre_X = pack_padded_sequence(pre_X, pre_length, batch_first=True)
    hyp_X = pack_padded_sequence(hyp_X, hyp_length, batch_first=True)
    return (pre_X, hyp_X, Y, pre_indices)

if __name__ == "__main__":
    pass
    # path = ['SNLI\data\snli_1.0\snli_1.0\snli_new_dev.csv',
    #         'SNLI\data\snli_1.0\snli_1.0\snli_new_test.csv',
    #         'SNLI\data\snli_1.0\snli_1.0\snli_new_train.csv']
    # vocab_save_path = 'SNLI\data\snli_1.0\snli_1.0\snil_vocab.txt'
    # generate_vocab(path, vocab_save_path)
    # clear_data_('SNLI\data\snli_1.0\snli_1.0\snli_1.0_train.txt', 'SNLI\data\snli_1.0\snli_1.0\snli_new_train.csv')

    # vocab = Vocab('SNLI\data\snli_1.0\snli_1.0\snil_vocab.txt')
    # dataset = DataSet('SNLI\data\snli_1.0\snli_1.0\snli_new_dev.csv', vocab)