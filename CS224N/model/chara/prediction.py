# -*- coding: utf-8 -*
import os
import torch
from data_helper import DATA_PATH, MODEL_PATH
from transformer import Transformer
from vocab import Vocab, VocabEntry
from data_helper import load_dict, text2id


class Prediction(object):
    def __init__(self):
        ''' init
        '''
        self.BATCH = 1
        self.model_dim = 512
        self.tgt_max_len = 300
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        self.load_model()

    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        # 加载词典
        src_vocab = VocabEntry('data/ask_chara.json', isAsk=True)
        tgt_vocab = VocabEntry('data/ans_chara.json')
        self.vocab = Vocab(src_vocab, tgt_vocab)
        # 加载模型
        self.model = Transformer(device=self.device,
                            src_vocab_size = len(self.vocab.src),
                            src_max_len = 400,
                            tgt_vocab_size = len(self.vocab.tgt),
                            tgt_max_len = self.tgt_max_len,
                            num_layers=3,
                            model_dim=self.model_dim,
                            num_heads=8,
                            ffn_dim=2048,
                            dropout=0.2)
        self.load_model_file(self.model)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, department, title, ask):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"department": "心血管科", "title": "心率为72bpm是正常的吗",
                                    "ask": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"answer": "心脏不舒服一般是..."}
        '''
        src = department + title + ask
        dep, dep_len = self.vocab.src.sent2ids(department)
        dep = [self.vocab.src.dep_id] + dep
        dep_len += 1
        tit, tit_len = self.vocab.src.sent2ids(title)
        tit = [self.vocab.src.tit_id] + tit
        tit_len += 1
        ask, ask_len = self.vocab.src.sent2ids(ask)
        ask = [self.vocab.src.ask_id] + ask + [self.vocab.src.eos_id]
        ask_len += 2
        
        src_x = dep + tit + ask
        src_len = dep_len + tit_len + ask_len
        src_len = [src_len]

        tgt = [self.vocab.tgt.sos_id]
        predict = self.model.predict(src_x, src_len, tgt)
        # predict = predict.squeeze_().tolist()
        result_list = list()
        for item in predict:
            if item != self.vocab.tgt.unk_id:
                result_list.append(self.vocab.tgt.id2text.get(item, self.vocab.tgt.unk))
        result = ''.join(result_list)
        return {'answer': result}
    
    def save_model(self, model):
        torch.save(model.state_dict(), 'data/model.pkl')
    
    def load_model_file(self, model):
        model.load_state_dict(torch.load('data/model.pkl'))


if __name__ == "__main__":
    Pre = Prediction()
    department = "心血管科"
    title = "心率为72bpm是正常的吗"
    ask = "最近不知道怎么回事总是感觉心脏不舒服..."
    print(Pre.predict(department, title, ask))