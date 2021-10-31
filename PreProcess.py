"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import os
import shutil
import cv2
import datetime


def dir_check(filepath, print_flag=True, empty_flag=False):
    """
    Empty all contents in the folder
    :param filepath: input file path
    :param empty_flag: empty flag
    :param print_flag: print flag
    :return: None
    """
    if os.path.exists(filepath) and empty_flag:
        del_file(filepath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if print_flag:
        print('%s folder has been emptied and recreated' % filepath)


def del_file(filepath):
    """
    Empty all contents in the folder
    :param filepath: input file path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def PreProcess(dataset):
    # Process the dataset
    if dataset == 'drive':  # drive dataset
        print('drive dataset is preprocessing')
        # input and output folder
        input_dir = ['Dataset\\DRIVE\\training', 'Dataset\\DRIVE\\test']
        output_dir = ['train', 'test']
        # Empty the output folder
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        num = 1  # image number
        for f in os.listdir(input_dir[0]):
            img_path = os.path.join(input_dir[0], f)  # image path
            img = cv2.imread(img_path)  # read the image
            cv2.imwrite(os.path.join(output_dir[0], '%d.png' % num), img)  # save the image
            num = num + 1
        for f in os.listdir(input_dir[1]):
            img_path = os.path.join(input_dir[1], f)  # image path
            img = cv2.imread(img_path)  # read the image
            cv2.imwrite(os.path.join(output_dir[1], '%d.png' % num), img)  # save the image
            num = num + 1
    elif dataset == 'PH2':  # PH2 dataset
        print('PH2 dataset is preprocessing')
        # input and output folder
        input_dir = 'Dataset\\PH2\\trainx'
        output_dir = ['train', 'test']
        # Empty the output folder
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        num = 1  # image number
        for f in os.listdir(input_dir):
            img_path = os.path.join(input_dir, f)  # image path
            img = cv2.imread(img_path)  # read the image
            if num <= 125:
                cv2.imwrite(os.path.join(output_dir[0], '%d.png' % num), img)  # save the image
            elif 125 < num <= 200:
                cv2.imwrite(os.path.join(output_dir[1], '%d.png' % (num - 125)), img)  # save the image
            num = num + 1
    elif dataset == 'Massachusetts':  # Massachusetts Buildings dataset
        print('Massachusetts Buildings dataset is preprocessing')
        # input and output folder
        input_dir = ['Dataset\\Massachusetts\\train', 'Dataset\\Massachusetts\\test']
        output_dir = ['train', 'test']
        # Empty the output floder
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        num = 1  # image number
        img_path_list = os.listdir(input_dir[0])  # image path
        img_path_list = img_path_list[0:round(len(img_path_list) * 0.7)]
        for path in img_path_list:
            img_path = os.path.join(input_dir[0], path)  # image path
            img = cv2.imread(img_path)  # read the image
            img = cv2.resize(img, (200, 200))  # resize
            cv2.imwrite(os.path.join(output_dir[0], '%d.png' % num), img)  # save the image
            num = num + 1
        img_path_list = os.listdir(input_dir[1])
        img_path_list = img_path_list[0:round(len(img_path_list) * 0.7)]
        for path in img_path_list:
            img_path = os.path.join(input_dir[1], path)  # image path
            img = cv2.imread(img_path)  # read the image
            img = cv2.resize(img, (200, 200))  # resize
            cv2.imwrite(os.path.join(output_dir[1], '%d.png' % num), img)  # save the image
            num = num + 1
    else:
        raise NameError('Input dataset does not exist')