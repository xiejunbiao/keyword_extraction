# coding=utf-8
import pickle
import os
import sys
import torch
import torch.optim as optim
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
from ChineseNER.pytorch.BilstmCrf import BiLstmCrf
from ChineseNER.pytorch.resultCal import calculate


class ModelBiLstmCrf(object):

    def __init__(self, data_path_, model_path, k=0):

        with open(os.path.join(data_path_, 'data.pkl'), 'rb') as inp:
            self.word2id = pickle.load(inp)
            self.id2word = pickle.load(inp)
            self.tag2id = pickle.load(inp)
            self.id2tag = pickle.load(inp)
            self.x_train = pickle.load(inp)
            self.y_train = pickle.load(inp)
            self.x_test = pickle.load(inp)
            self.y_test = pickle.load(inp)
            self.x_valid = pickle.load(inp)
            self.y_valid = pickle.load(inp)
        self.model_path = os.path.join(model_path, 'model_lstm_crf.pkl')
        self.k = k
        self.max_len = 60
        self.model = None

    def train_model(self):
        """
        训练并且返回一个模型
        :return:
        """
        if os.path.exists(self.model_path):

            self.model = torch.load(self.model_path)
            return '已有模型'

        else:
            result = self._train_fun()
            self.model = torch.load(self.model_path)
            return result

    def _train_fun(self):
        """

        @return:
        @rtype:
        """
        start_tag = "<START>"
        stop_tag = "<STOP>"
        embedding_dim = 100
        hidden_dim = 200
        epochs = 5

        self.tag2id[start_tag] = len(self.tag2id)
        self.tag2id[stop_tag] = len(self.tag2id)
        model = BiLstmCrf(len(self.word2id) + 1, self.tag2id, embedding_dim, hidden_dim)
        # embedding_dim:输入' x '中预期的特性数量
        # hidden_size: 隐藏状态“h”的特性数
        zhun = 0
        zhao = 0
        f = 0

        optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        for epoch in range(epochs):
            index = 0
            for sentence, tags in zip(self.x_train, self.y_train):
                index += 1
                model.zero_grad()

                sentence = torch.tensor(sentence, dtype=torch.long)
                tags = torch.tensor([self.tag2id[t] for t in tags], dtype=torch.long)

                loss = model.neg_log_likelihood(sentence, tags)

                loss.backward()
                optimizer.step()
                if index % 300 == 0:
                    if self.k != 0:
                        print("epoch_{}".format(self.k), epoch, "index", index)
                    else:
                        print("epoch", epoch, "index", index)
        entityres = []
        entityall = []
        for sentence, tags in zip(self.x_test, self.y_test):
            sentence = torch.tensor(sentence, dtype=torch.long)
            score, predict = model(sentence)
            entityres = calculate(sentence, predict, self.id2word, self.id2tag, entityres)
            entityall = calculate(sentence, tags, self.id2word, self.id2tag, entityall)
        jiaoji = [i for i in entityres if i in entityall]
        if len(jiaoji) != 0:
            zhun = float(len(jiaoji)) / len(entityres)
            zhao = float(len(jiaoji)) / len(entityall)
            f = (2 * zhun * zhao) / (zhun + zhao)
            print("test:")
            print("准确率:", zhun)
            print("召回率:", zhao)
            print("f:", f)
        else:
            print("zhun:", 0)

        torch.save(model, self.model_path)
        print("model has been saved")
        return {'准确率：': zhun, '召回率：': zhao, 'f:': f}

    def x_padding(self, words):

        # 通过多个索引同时获取到对应的值
        ids = list(self.word2id[[w for w in words if w in self.word2id.index]])

        # 如果一句话过长就截取前max_len个字符
        if len(ids) >= self.max_len:
            return ids[:self.max_len]

        # 如果不够max_len个字符就用0填补
        ids.extend([0] * (self.max_len - len(ids)))
        return torch.tensor(ids)

    def predict(self, input_txt):
        """
        @param input_txt:
        @type input_txt:
        @return:
        @rtype:
        """
        input_txt = self.x_padding(input_txt)
        # input_txt = prepare_sequence(input_txt, self.word2id)
        # input_txt = torch.tensor(input_txt, dtype=torch.long)
        # model = self.train_model()
        score, predict0 = self.model(input_txt)
        # print(score, predict0)
        entityres = calculate(input_txt, predict0, self.id2word, self.id2tag)
        # print(entityres)
        # entityall = calculate(input_txt, tags, self.id2word, self.id2tag)
        word_list = []
        while entityres:
            word = entityres.pop()
            words = ''
            for each_word in word[:-1]: words = words + each_word.split('/')[0]
            word_list.append(words)
        return list(set(word_list))


def format_data(data):
    data_list = []
    for sent in data:
        data_list.append(str(sent).replace('\n', '').replace('\n', ''))


if __name__ == '__main__':
    # txt = 'ABC柔棉纤薄卫生巾日用18片/包青海湖鲜牛奶1L洗衣液'
    # txt = '青海湖鲜牛奶1L洗衣液ABC柔棉纤薄卫生巾日用18片/包'
    # txt = '牛奶味的洗衣液'
    txt = 'Elleair卷筒卫生纸柔软亲肤型10'
    data_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation\\data0'
    # 此路径为预处理之后的数据路径
    # model_save_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\pytorch\\model'  # 模型的路径
    model_save_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation\\data0'
    # 模型的路径

    # txt = sentence = torch.tensor(x_test[0], dtype=torch.long)
    # test_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_test\\all_data.txt'
    model = ModelBiLstmCrf(data_path, model_save_path)
    model.train_model()
    result = model.predict(txt)
    # f = open(test_path, 'r', encoding='utf-8')

    print(result)
