import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch
import numpy as np

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
                vocab.add(word)
            s = itor['sentence2']
            if not isinstance(s, str):
                s = '_pad_'
            words = s.split()
            for word in words:
                vocab.add(word)
    # 保存单词表
    vocab.discard('_pad_')
    with open(vocab_save_path, mode='w+', encoding='ascii') as file:
        for word in vocab:
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
        self.vocab = {}
        self._pad_ = 0
        self._gap_ = 1
        self.vocab['_pad_'] = self._pad_
        self.vocab['::'] = self._gap_
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
            if word not in self.vocab:
                self.vocab[word] = begin
                begin += 1
        return self.vocab
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, word):
        return self.vocab.get(word, self._pad_)
    
    def sents2indexs(self, sents, max_length=None):
        """ 句子转index
        :param sents ([str])
        :param max_length (int): 句子padding的长度
        :return indexs ndarray(n, max_length)
        """
        indexs = []
        for sent in sents:
            indexs.append(self.sent2index(sent, max_length))
        return indexs
    
    def sent2index(self, sent:str, max_length=None):
        """ sent(str) 转 ndarray(max_length)
        """
        if max_length is None:
            # 不padding
            sent = [self.__getitem__(word) for word in sent.split()]
        else:
            # padding
            sent = sent.split()
            if (len(sent) > self.max_l):
                self.max_l = len(sent)
            if len(sent) > max_length:
                sent = [self.__getitem__(word) for word in sent[:max_length]]  
            else:
                sent = [self.__getitem__(word) for word in sent] + [self._pad_]* (max_length - len(sent))
        return sent


class DataSet(Dataset):
    def __init__(self, path, vocab: Vocab, max_length=None):
        super(DataSet, self).__init__()
        self.vocab = vocab
        self.data_Y = None
        self.data_X_1 = None
        self.data_X_2 = None
        self.load_data(path, max_length)

    def __getitem__(self, index):
        return (torch.tensor(self.data_X_1[index]), torch.tensor(self.data_X_2[index]), torch.tensor(self.data_Y[index]))

    def __len__(self):
        return len(self.data_Y)

    def load_data(self, path, max_length=None):
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
        self.data_X_1 = self.vocab.sents2indexs(data_X_1, max_length)
        self.data_X_2 = self.vocab.sents2indexs(data_X_2, max_length)
        return (self.data_X_1, self.data_X_2, self.data_Y)
        




if __name__ == "__main__":
    # path = ['SNLI\data\snli_1.0\snli_1.0\snli_new_dev.csv',
    #         'SNLI\data\snli_1.0\snli_1.0\snli_new_test.csv',
    #         'SNLI\data\snli_1.0\snli_1.0\snli_new_train.csv']
    # vocab_save_path = 'SNLI\data\snli_1.0\snli_1.0\snil_vocab.txt'
    # generate_vocab(path, vocab_save_path)
    # clear_data_('SNLI\data\snli_1.0\snli_1.0\snli_1.0_train.txt', 'SNLI\data\snli_1.0\snli_1.0\snli_new_train.csv')

    vocab = Vocab('SNLI\data\snli_1.0\snli_1.0\snil_vocab.txt')
    dataset = DataSet('SNLI\data\snli_1.0\snli_1.0\snli_new_dev.csv', vocab)