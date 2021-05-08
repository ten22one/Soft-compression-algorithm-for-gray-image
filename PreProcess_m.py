"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import os
import shutil
import cv2


def dir_check(filepath, print_flag=True, empty_flag=False):
    if os.path.exists(filepath) and empty_flag:
        del_file(filepath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if print_flag:
        print('Output folder %s has been cleaned and created' % filepath)


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


if __name__ == '__main__':

    print('Please input the dataset name, you can choose: DRIVE PH2：', end='')
    dataset = input()

    if dataset == 'DRIVE':
        input_dir = ['Dataset\\DRIVE\\training', 'Dataset\\DRIVE\\test']
        output_dir = ['train', 'test']
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        num = 1
        for f in os.listdir(input_dir[0]):
            img_path = os.path.join(input_dir[0], f)
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_dir[0], '%d.png' % num), img)
            num = num + 1
        for f in os.listdir(input_dir[1]):
            img_path = os.path.join(input_dir[1], f)
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_dir[1], '%d.png' % num), img)
            num = num + 1
    elif dataset == 'PH2':
        input_dir = 'Dataset\\PH2\\trainx'
        output_dir = ['train', 'test']
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)

        num = 1
        for f in os.listdir(input_dir):
            img_path = os.path.join(input_dir, f)
            img = cv2.imread(img_path)
            if num <= 125:
                cv2.imwrite(os.path.join(output_dir[0], '%d.png' % num), img)
            elif 125 < num <= 200:
                cv2.imwrite(os.path.join(output_dir[1], '%d.png' % (num - 125)), img)
            num = num + 1
