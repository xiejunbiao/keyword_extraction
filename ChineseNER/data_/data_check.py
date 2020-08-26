# -*- coding: utf-8 -*-
"""
Create Time: 2020/8/25 11:55
Author: xiejunbiao
"""
import os
import sys
from progress.bar import Bar
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
from ChineseNER.pytorch.train import ModelBiLstmCrf
import pandas as pd


class CheckData(object):
    def __init__(self, data_path, model_save_path):
        self.model = ModelBiLstmCrf(data_path, model_save_path)
        self.model.train_model()

    def check_start(self, original_data, result_path):
        data_final = []
        data = self.get_data(original_data)
        print(data['data'][16884:16890])
        with Bar('Processing', max=len(data['data'])) as bar:
            for each in data['data']:
                try:
                    bar.next()
                    result = self.model.predict(each['spu_name'][:59])
                    print('名称：', each, 'result:', ','.join(result))
                    if len(result) == 1 and each['cls'] not in result:
                        data_final.append({'spu_name': each['spu_name'],
                                           'predict': ','.join(result[0]),
                                           'cls': each['cls'],
                                           'type': 1})
                    elif len(result) == 2 and each['cls'] not in result:
                        data_final.append({'spu_name': each['spu_name'],
                                           'predict': ','.join(result),
                                           'cls': each['cls'],
                                           'type': 2})

                    elif len(result) >= 3 and each['cls'] not in result:
                        data_final.append({'spu_name': each['spu_name'],
                                           'predict': ','.join(result),
                                           'cls': each['cls'],
                                           'type': '3-'})

                    elif len(result) == 0 and each['cls'] not in result:
                        data_final.append({'spu_name': each['spu_name'],
                                           'predict': ','.join(result),
                                           'cls': each['cls'],
                                           'type': 0})
                    else:
                        data_final.append({'spu_name': each['spu_name'],
                                           'predict': ','.join(result),
                                           'cls': each['cls'],
                                           'type': -1})
                except Exception as e:
                    print(e, each)
        df = pd.DataFrame(data_final)
        df.to_csv(result_path)

    @staticmethod
    def get_data(original_data):
        data = {'data': [], 'valid': []}
        f = open(original_data, 'r', encoding='utf-8')

        for line in f.readlines():
            each_data = line.replace('\n', '').replace(' ', '').replace(',', '').split('\t')
            if len(each_data) == 3:
                data['data'].append({'spu_name': ','.join(each_data[:2]), 'cls': each_data[2]})
            else:
                data['valid'].append(each_data)
        return data


if __name__ == '__main__':
    data_path1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation\\data0'
    # 此路径为预处理之后的数据路径
    # model_save_path = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\pytorch\\model'  # 模型的路径
    model_save_path1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER\\data_\\fold_cross_validation\\data0'
    original_data1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER' \
                     '\\data_\\fold_cross_validation\\data_original_all_o.txt'
    result_path1 = 'E:\\Document\\project\\keyword_extraction\\ChineseNER' \
                   '\\data_\\fold_cross_validation\\result_data.csv'
    cd = CheckData(data_path1, model_save_path1)
    cd.check_start(original_data1, result_path1)
