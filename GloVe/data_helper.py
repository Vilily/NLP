""" 数据处理部分
@Author: Bao Wenjie
@Date: 2020/8/6
@Email: bwj_678@qq.com
"""
import json

class Vocab:
    """ 词典类
    """
    def __init__(self, path):
        super().__init__()
        self.word2id = {}
        self.id2word = {}
        if(path != None and path != ''):
            self.load_data(path)
    
    def generate_vocab(self, src_path, tar_path):
        """ 生成词典文件
        @param src_path(str): 源路径
        """
        fp = open(src_path, 'r', encoding='utf-8')
        self.word2id = {}
        i = 1
        for line in fp.readlines():
            line = line.split(' ')
            for word in line:
                if(word not in self.word2id):
                    word = word.strip('\n').strip()
                    if(word.isalpha()):
                        self.word2id[word] = i
                        i += 1
        tmp = json.dumps(self.word2id)
        with open(tar_path, 'w+', encoding='utf-8') as fp:
            fp.write(tmp)

    def load_data(self, path):
        """ 加载词典文件
        """
        data = None
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.loads(fp.read())
        self.word2id = data
        for item in self.word2id.items():
            self.id2word[item[1]] = item[0]
        self.id2word[0] = '_unk_'
        self.word2id['_unk_'] = 0
    
    def __len__(self):
        """ len(Vocab) 实现
        """
        return len(self.word2id)
    
    def __getitem__(self, word):
        """ Vocab[word]实现
        """ 
        return self.word2id.get(word, -1)

class DateHelper:
    """ 数据处理类
    """
    def __init__(self, path, vocab):
        """ 初始化
        """
        super().__init__()
        self.vocab = vocab
        self.copora = None
        self.load_data(path)

    def load_data(self, path):
        """ 读取语料库
        @param path(str): 路径
        @return copora(List[str]): 语料库
        """
        with open(path, 'r', encoding='utf-8') as fp:
            self.copora = fp.readlines()
    
    def __len__(self):
        return len(self.copora)
    
    def get(self, begin, batch=1):
        """ 获取数据
        """
        pass

    def dealWithSentence(self, sentence):
        """ 从一句话中生成共现矩阵
        """
        pass



if __name__ == '__main__':
    src_path = 'GloVe\\data.txt'
    tar_path = 'GloVe\\vocab.json'
    vocab = Vocab(tar_path)
    # vocab.generate_vocab(src_path, tar_path)
