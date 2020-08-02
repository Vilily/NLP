''' BLEU (BiLingual Evaluation Understudy)
@Author: Bao Wenjie
@Date: 2020/8/2
@Email: bwj_678@qq.com
'''
import numpy as np
from nltk import bleu
from nltk.translate import bleu_score

class BLEU():
    def __init__(self, n_gram=1):
        super().__init__()
        self.n_gram = n_gram

    def evaluate(self, candidate, reference):
        ''' 计算BLEU值
        @param candidates [str]: 机器翻译的句子
        @param references [str]: 参考的句子
        @param bleu: BLEU值
        '''
        count = np.zeros(self.n_gram)
        count_clip = np.zeros(self.n_gram)
        p = np.zeros(self.n_gram)
        bleu = 0
        BP = 1
        for i in range(self.n_gram):
            count[i], n_grams = self.extractNgram(candidate, i + 1)
            count_clip[i] = self.countClip(reference, i + 1, n_grams)
        rc = len(reference)/len(candidate)
        if(rc >= 1):
            BP = np.exp(1 - rc)
        p = count_clip / count
        return BP * np.exp(np.average(np.log(p)))

    def extractNgram(self, candidate, n):
        ''' 抽取出n-gram
        @param candidates [str]: 机器翻译的句子
        @param n int: n-garm值
        @return count int: n-garm个数
        @return n_grams set(): n-grams 
        '''
        count = 0
        n_grams = set()
        if(len(candidate) - n + 1 > 0):
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(candidate[i:i+n])
            n_grams.add(n_gram)
        return (count, n_grams)
    
    def countClip(self, reference, n, n_grams):
        ''' 计数references中有多少n_grams
        @param references [str]: 参考译文
        @param n int: n-gram的值
        @param n_grams set(): n-grams
        '''
        count = 0
        for i in range(len(reference) - n + 1):
            if(' '.join(reference[i:i+n]) in n_grams):
                count += 1
        return count


if __name__ == '__main__':
    bleu_ = BLEU(4)
    candidate = 'the the the the the the the'
    candidate = candidate.split()
    reference = 'The cat is on the mat'
    reference = reference.split()
    print(bleu_.evaluate(candidate, reference))
    print(bleu_score.corpus_bleu([[reference]], [candidate]))