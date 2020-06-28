import jieba
import pandas as pd
import json


def generate_dict(src_path, store_path_ask, store_path_ans):
    text2idAsk = {}
    text2idAns = {}
    data = pd.read_csv(src_path)
    i = 4
    j = 4
    for ind, row in data.iterrows():
        sent = row['department']
        textline = jieba.lcut(sent)
        for word in textline:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['title']
        textline = jieba.lcut(sent)
        for word in textline:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['ask']
        textline = jieba.lcut(sent)
        for word in textline:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['answer']
        textline = jieba.lcut(sent)
        for word in textline:
            if(word not in text2idAns):
                text2idAns[word] = j
                j += 1
    print(len(text2idAns),len(text2idAsk))
    json_data_ask = json.dumps(text2idAsk)
    json_data_ans = json.dumps(text2idAns)
    with open(store_path_ask, 'w+') as file:
        file.write(json_data_ask)
    with open(store_path_ans, 'w+') as file:
        file.write(json_data_ans)



def generate_little_dict(src_path, store_path):
    text2id = {}
    data = pd.read_csv(src_path)
    for ind, row in data.iterrows():
        # sent = row['department']
        # textline = jieba.lcut(sent)
        # for word in textline:
        #     if(word not in text2id):
        #         text2id[word] = 1
        #     text2id[word] += 1
        # sent = row['title']
        # textline = jieba.lcut(sent)
        # for word in textline:
        #     if(word not in text2id):
        #         text2id[word] = 1
        #     text2id[word] += 1
        # sent = row['ask']
        # textline = jieba.lcut(sent)
        # for word in textline:
        #     if(word not in text2id):
        #         text2id[word] = 1
        #     text2id[word] += 1
        sent = row['answer']
        textline = jieba.lcut(sent)
        for word in textline:
            if(word not in text2id):
                text2id[word] = 1
            text2id[word] += 1
    text2id = sorted(text2id.items(), key=lambda item:item[1], reverse=True)
    i = 4
    data = {}
    for item in text2id[:20000]:
        data[item[0]] = i
        i += 1
    json_data = json.dumps(data)
    with open(store_path, 'w+') as file:
        file.write(json_data)

def splitData(src_path, store_path):
    data = pd.read_csv(src_path)
    pd_dict = {}
    for i in range(data.shape[0]):
        row = data.loc[i]
        depart = row['department']
        if(depart in pd_dict):
            pd_dict[depart] = pd_dict[depart].append(row)
        else:
            pd_dict[depart] = pd.DataFrame(columns=('department', 'title', 'ask', 'answer'))
            pd_dict[depart] = pd_dict[depart].append(row, ignore_index=True)
    
    for key in pd_dict.keys():
        print(pd_dict[key].shape[0])
        pd_dict[key].to_csv(store_path + key + '.csv', index=0)

def generate_dict_chara(src_path, store_path_ask, store_path_ans):
    text2idAsk = {}
    text2idAns = {}
    data = pd.read_csv(src_path)
    i = 7
    j = 4
    for ind, row in data.iterrows():
        sent = row['department']
        for word in sent:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['title']
        for word in sent:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['ask']
        for word in sent:
            if(word not in text2idAsk):
                text2idAsk[word] = i
                i += 1
        sent = row['answer']
        for word in sent:
            if(word not in text2idAns):
                text2idAns[word] = j
                j += 1
    print(len(text2idAns),len(text2idAsk))
    json_data_ask = json.dumps(text2idAsk)
    json_data_ans = json.dumps(text2idAns)
    with open(store_path_ask, 'w+') as file:
        file.write(json_data_ask)
    with open(store_path_ans, 'w+') as file:
        file.write(json_data_ans)

if __name__ == '__main__':
    # splitData('data\\input\\train.csv', 'data\\input\\')
    # generate_little_dict('data\\input\\train.csv', 'data\\input\\ans.json')
    generate_dict_chara('data\\input\\train.csv', 'data\\input\\ask_chara.json', 'data\\input\\ans_chara.json')
