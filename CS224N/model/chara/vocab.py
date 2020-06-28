import pandas as pd
import json
import os

class VocabEntry(object):
    def __init__(self, file_path, isAsk=False):
        ''' 初始化字典入口类
        @param file_path: 字典路径
        '''
        self.isAsk = isAsk
        self.text2id = None
        self.id2text = None
        self.file_path = file_path
        self.pad = '_pad_'
        self.unk = '_unk_'
        self.sos = '_sos_'
        self.eos = '_eos_'
        self.dep = '_dep_'
        self.tit = '_tit_'
        self.ask = '_ask_'
        self.unk_id = 1
        self.pad_id = 0
        self.sos_id = 2
        self.eos_id = 3
        self.dep_id = 4
        self.tit_id = 5
        self.ask_id = 6
        self.load_dict(self.file_path)
    
    def __len__(self):
        ''' 返回Vocab长度
        @return len: 总的词汇长度
        '''
        return len(self.text2id)
    
    def __getitem__(self, word):
        ''' 返回text对应的id，不存在则返回 '_unk_'
        @param word: str, 查询的文本
        @param id: int, 对应的id
        '''
        return self.text2id.get(word, self.unk_id)
    
    def __contains__(self, word):
        ''' word词语是否在词典中
        @param contains(bool): 是否存在
        '''
        return word in self.text2id
    
    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)
    

    def load_dict(self, dictFile):
        ''' 加载字典
        @param: 字典路径
        @return text2id: dict, 根据文本查找id的字典
        @return id2text: dict, 根据id查找文本的字典
        '''
        if not os.path.exists(dictFile):
            print('[ERROR] load_dict failed! | The params {}'.format(dictFile))
            return None
        with open(dictFile, 'r', encoding='UTF-8') as df:
            dictF = json.load(df)
        self.text2id, self.id2text = dict(), dict()
        count = 4
        if(self.isAsk is True):
            count = 7
        for key, value in dictF.items():
            self.text2id[key] = count
            self.id2text[count] = key
            count += 1
        self.text2id[self.unk] = self.unk_id
        self.text2id[self.sos] = self.sos_id
        self.text2id[self.eos] = self.eos_id
        self.text2id[self.pad] = self.pad_id
        self.id2text[self.unk_id] = self.unk
        self.id2text[self.sos_id] = self.sos
        self.id2text[self.eos_id] = self.eos
        self.id2text[self.pad_id] = self.pad
        if(self.isAsk is True):
            self.text2id[self.dep] = self.dep_id
            self.text2id[self.tit] = self.tit_id
            self.text2id[self.ask] = self.ask_id
            self.id2text[self.dep_id] = self.dep
            self.id2text[self.tit_id] = self.tit
            self.id2text[self.ask_id] = self.ask
        return self.text2id, self.id2text

    def id2text(self, id):
        return self.id2text[id]
    
    def text2id(self, text):
        return self.text2id[text]

    def sent2ids(self, sents):
        ''' 将一个句子str转成 List[id]
        @param sents (str): 句子
        @return word_ids ([list[int]): 句子的表征
        @return text_len (int): 句子长度
        '''
        ids_list = [self.text2id.get(w, self.unk_id) for w in sents]
        text_len = len(ids_list)
        return (ids_list, text_len)

    def ids2sents(self, word_ids):
        ''' 将ids转成句子
        @param word_ids List[int]: id列表
        @return sents (str): 句子
        '''
        words = [self.id2text.get(w_id, self.unk) for w_id in word_ids]
        return ''.join(self.ids2words(word_ids))

    def id2word(self, wid):
        return self.id2text.get(wid, self.unk)
    
    def word2id(self, word):
        return self.text2id.get(word, self.unk_id)

class Vocab(object):
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab