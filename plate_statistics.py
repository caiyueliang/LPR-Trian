# encoding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         '港', '学', '使', '警', '澳', '挂', '军', '北', '南', '广',
         '沈', '兰', '成', '济', '海', '民', '航', '空',
         ]

CHARS_DICT = {char.decode("utf-8"): i for i, char in enumerate(CHARS)}


def statistics(label_path):
    # 导入车牌数据
    str_lists = []

    file1 = open(label_path)
    lines = file1.readlines()

    for line in lines:
        plate = line.replace('\r', '').replace('\n', '').split(':')[-1].decode('utf-8')

        # if list(plate)[1] == '0':
        #     print(line)

        for i, char in enumerate(list(plate)):
            if i >= len(str_lists):
                str_lists.append([])

            str_lists[i].append(char)

    for str_list in str_lists:
        total_count = len(str_list)
        for char in CHARS:
            char_count = str_list.count(char.decode('utf-8'))
            if char_count != 0:
                print '%s: %4d, %.6f  ' % (char.decode('utf-8'), char_count, float(char_count)/total_count),
        print('\n')
    return


if __name__ == "__main__":
    label_path = "../Data/car_recognition/train/labels_normal.txt"

    statistics(label_path)
