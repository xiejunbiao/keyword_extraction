# -*- coding: utf-8 -*-
"""
Create Time: 2020/8/20 17:55
Author: xiejunbiao
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
from ChineseNER.pytorch.train import ModelBiLstmCrf
from threading import Thread, Lock


def fun_(data_path, k):
    txt = 'Elleair 卷筒卫生纸柔软亲肤型10'
    model = ModelBiLstmCrf(data_path, data_path, k)
    evaluation = model.train_model()
    result = model.predict(txt)
    print('第{}个线程的结果过为{}, \t 评价结果过为{}'.format(k, result, evaluation))


def start(data_path):
    for i in range(5):
        data_path_ = os.path.join(data_path, "data{}".format(i+1))
        Thread(target=fun_, args=(data_path_, i+1)).start()
        print('第{}个线程已启动'.format(i+1))


if __name__ == '__main__':

    data_path1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation'  # 此路径为预处理之后的数据路径
    model_save_path1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation'  # 模型的路径
    start(data_path1)
