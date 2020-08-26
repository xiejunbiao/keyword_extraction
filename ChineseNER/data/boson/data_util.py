# -*- coding: UTF-8 -*-
import collections
import os
import sys
import pickle
import codecs
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# pathDir = os.path.dirname(__file__)
# curPath = os.path.abspath(pathDir)
# rootPath = os.path.split(curPath)[0]
# sys.path.append(os.path.split(rootPath)[0])
# sys.path.append(rootPath)
from preprocess_data import clear_and_format_data2


class PreProcess(object):
    def __init__(self):
        self.max_len = 60

    def flatten(self, list_args):
        result = []
        for el in list_args:
            # collections.Iterable是一种迭代类型
            # isinstance(x1,type)判断x1的类型是否和type相同
            if isinstance(list_args, collections.Iterable) and not isinstance(el, str):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def data2pkl(self, path, save_path):
        max_len = self.max_len
        datas = list()  # 样本
        labels = list()  # 样本标签
        # linedata = list()
        # linelabel = list()
        tags = set()
        input_data = codecs.open(os.path.join(path, 'spu_name_wordtagsplit.txt'), 'r', 'utf-8')
        # input_data = codecs.open('./data_t/spu_name_wordtagsplit.txt', 'r', 'utf-8')
        # input_data = codecs.open('./word_small.txt', 'r', 'utf-8')
        # input_data = codecs.open('./wordtagsplit.txt', 'r', 'utf-8')

        # 按行读取并且按行分析样本
        for line in input_data.readlines():
            # split 划分默认为空格
            line = line.split()
            linedata = []
            linelabel = []
            num_noto = 0
            for word in line:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(word[1])
                tags.add(word[1])
                if word[1] != 'O':
                    num_noto += 1
            if num_noto != 0:
                datas.append(linedata)
                labels.append(linelabel)

        input_data.close()
        # print(len(datas), tags)
        # print(len(labels))
        all_words = self.flatten(datas)

        # 将所有的字转换为序列 格式如下
        """
        0   牛
        1   肉
        2   丸
        。。。   
        """
        sr_allwords = pd.Series(all_words)
        # print(sr_allwords)

        # 统计序列中的字的个数 返回一个序列返回每个字的数量
        sr_allwords = sr_allwords.value_counts()
        # print(sr_allwords)

        # 获取所有字的一个集合（不重复）
        set_words = sr_allwords.index
        # print(set_words)

        set_ids = range(1, len(set_words)+1)
        # print(set_ids)

        # 将集合中的元素转成列表
        tags = [i for i in tags]
        tag_ids = range(len(tags))
        # print(tag_ids)

        # 生成两个序列
        word2id = pd.Series(set_ids, index=set_words)
        # print(word2id)

        id2word = pd.Series(set_words, index=set_ids)
        # print(id2word)

        tag2id = pd.Series(tag_ids, index=tags)
        id2tag = pd.Series(tags, index=tag_ids)

        # 添加一个不知道的索引值为 len(word2id)+1
        word2id["unknow"] = len(word2id)+1
        # print(word2id)

        # 定义一句话的长度为60

        def x_padding(words):
            # 通过多个索引同时获取到对应的值
            ids = list(word2id[words])

            # 如果一句话过长就截取前max_len个字符
            if len(ids) >= max_len:
                return ids[:max_len]

            # 如果不够max_len个字符就用0填补
            ids.extend([0]*(max_len-len(ids)))
            return ids

        def y_padding(tags1):
            ids = list(tag2id[tags1])
            if len(ids) >= max_len:
                return ids[:max_len]
            ids.extend([0]*(max_len-len(ids)))
            return ids

        #
        df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
        df_data['x'] = df_data['words'].apply(x_padding)
        df_data['y'] = df_data['tags'].apply(y_padding)
        x = np.asarray(list(df_data['x'].values))
        y = np.asarray(list(df_data['y'].values))
        path_save = os.path.join(save_path, "data0\\data.pkl")

        # kf = KFold(n_splits=5)
        # i = 1
        # for train_index, test_index in kf.split(x):
        #       path_save = os.path.join(save_path, "data{}\\data.pkl".format(i), )
        #     i += 1
        #     print(path_save)
        #     train_x, train_y = x[train_index], y[train_index]
        #     x_test, y_test = x[test_index], y[test_index]
        x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.01, random_state=43)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.01, random_state=43)

        with open(path_save, 'wb') as outp:
            # with open('../Bosondata1.pkl', 'wb') as outp:
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(tag2id, outp)
            pickle.dump(id2tag, outp)
            pickle.dump(x_train, outp)
            pickle.dump(y_train, outp)
            pickle.dump(x_test, outp)
            pickle.dump(y_test, outp)
            pickle.dump(x_valid, outp)
            pickle.dump(y_valid, outp)
        print('** Finished saving the data.')

    @staticmethod
    def origin2tag(path):

        input_data = codecs.open(os.path.join(path, 'spu_name_origindata.txt'), 'r', 'utf-8')
        output_data = codecs.open(os.path.join(path, 'spu_name_wordtag.txt'), 'w', 'utf-8')
        # input_data = codecs.open('./spu_name_origindata.txt', 'r', 'utf-8')
        # output_data = codecs.open('./spu_name_wordtag.txt', 'w', 'utf-8')
        for line in input_data.readlines():
            line = line.strip()
            i = 0
            while i < len(line):
                if line[i] == '{':
                    i += 2
                    temp = ""
                    while line[i] != '}':
                        temp += line[i]
                        i += 1
                    i += 2
                    word = temp.split(':')
                    sen = word[1]
                    try:
                        output_data.write(sen[0]+"/B_"+word[0]+" ")
                    except Exception as e:
                        pass
                        # print('sen:', sen)
                        # print('line:', line)
                        # print('word:', word)
                    for j in sen[1:len(sen)-1]:
                        output_data.write(j+"/M_"+word[0]+" ")
                    output_data.write(sen[-1]+"/E_"+word[0]+" ")
                else:
                    output_data.write(line[i]+"/O ")
                    i += 1
            output_data.write('\n')
        input_data.close()
        output_data.close()

    @staticmethod
    def tagsplit(path):
        # with open('./data_t/xwj_wordtag.txt', 'rb') as inp:
        inp = open(os.path.join(path, 'spu_name_wordtag.txt'), 'r', encoding='utf-8')
        # inp = open('./spu_name_wordtag.txt', 'r', encoding='utf-8')
        # 将txt中的文本一股脑的全部读出
        # texts = inp.read().decode('utf-8')
        sentences = inp.readlines()
        # print(texts)

        # 此处为何要划分
        # sentences = re.split("[，。！？、‘’“”（）]/[O]", texts)
        # sentences = re.split('[，。！？、‘’“”（）]/[O]'.decode('utf-8'), texts)
        # output_data = codecs.open('./data_t/spu_name_wordtagsplit.txt', 'w', 'utf-8')
        output_data = codecs.open(os.path.join(path, 'spu_name_wordtagsplit.txt'), 'w', 'utf-8')
        for sentence in sentences:
            if sentence != " ":
                output_data.write(sentence.strip()+'\n')
        output_data.close()


def get_predata(data_name, path_save):
    """
    data_name：原数据路径+名称
    path_save：处理后的数据保存路径
    """
    # org_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data\\boson'
    # path_root = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data\\boson\\data_t'
    # path_save = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data'
    clear_and_format_data2(data_name, path_save)  # 数据清洗和数据格式化 在路径path_save下产生文件 spu_name_origindata.txt
    print('----------------数据清洗完成-------------')
    preprocess = PreProcess()

    preprocess.origin2tag(path_save)  # 在路径path_save下产生文件spu_name_wordtag.txt

    preprocess.tagsplit(path_save)  # 在路径path_save下产生文件spu_name_wordtagsplit.txt

    preprocess.data2pkl(path_save, save_path=path_save)  # 在路径path_save下产生文件 data.pkl
    print('----------------数据处理完成-------------')


if __name__ == '__main__':
    path_root = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\data_original_all.txt'
    path_sa = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation'
    get_predata(path_root, path_sa)
