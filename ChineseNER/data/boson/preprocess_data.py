import os
# {{product_name:浙江在线杭州}}
# if __name__ == '__main__':
# path = './data.txt'
# save_path = './spu_name_origindata.txt'


def clear_and_format_data1(data_name, save_path):
    """
    原数据格式为：
    商品名                 标签
    舒洁双层抽取纸4包装	纸
    舒洁精品3层卷筒卫生纸260节*10	卫生纸
    舒洁无香湿巾单包装10片	纸
    清风原木纯品面纸3包装	面纸
    舒洁迪斯尼软抽3包装	软抽
    Frosch芦荟洗衣液（瓶装）1500ml	洗衣液
    """
    f = open(data_name, 'r', encoding='utf-8')
    data_ = f.readlines()
    f.close()
    f_save = open(os.path.join(save_path, 'spu_name_origindata.txt'), 'w', encoding='utf-8')
    for line in data_:
        each_data = line.replace('\n', '').replace(' ', '').replace('{', '').replace('}', '').split('\t')
        print(each_data)
        if len(each_data) == 2 and each_data[1] in each_data[0]:
            str_temp = each_data[0].replace(each_data[1], '{{spu_name:%s}}' % each_data[1])
            f_save.write(str_temp + '\n')
            print(str_temp)
        else:
            f_save.write(each_data[0] + '\n')


def clear_and_format_data2(data_name, save_path):
    """
        原数据格式为：
        商品名              二级类别  标签
        舒洁双层抽取纸4包装	xxxx     纸
        舒洁精品3层卷筒卫生纸260节*10	xxxx     卫生纸
        舒洁无香湿巾单包装10片	xxxx     纸
        清风原木纯品面纸3包装	 xxxx    面纸
        舒洁迪斯尼软抽3包装	xxxx     软抽
        Frosch芦荟洗衣液（瓶装）1500ml   xxxx	洗衣液
        """
    f = open(data_name, 'r', encoding='utf-8')
    data_ = f.readlines()
    f.close()
    f_save = open(os.path.join(save_path, 'spu_name_origindata.txt'), 'w', encoding='utf-8')
    for line in data_:
        each_data = line.replace('\n', '').replace(' ', '').replace('{', '').replace('}', '').split('\t')
        # print(each_data)
        if len(each_data) == 3 and each_data[2] in ','.join(each_data[:2]):
            str_temp = ','.join(each_data[:2]).replace(each_data[2], '{{spu_name:%s}}' % each_data[2])
            f_save.write(str_temp + '\n')
            # print(str_temp)
        elif len(each_data) == 2 and each_data[1] in each_data[0]:
            str_temp = each_data[0].replace(each_data[1], '{{spu_name:%s}}' % each_data[1])
            f_save.write(str_temp + '\n')
            # print(str_temp)
        else:
            f_save.write(each_data[0] + '\n')
