"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
PreProcess.py - Generate the training set and testing set
"""


import os
import shutil
import cv2
import datetime
import struct
import numpy as np


def cifar10_img(file_dir, export_dir):
    loc_1, loc_2 = export_dir
    for i in range(1, 6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_path = os.path.join(loc_1, str(data_dict[b'labels'][j]))
            dir_check(img_path, print_flag=False, empty_flag=False)
            img_name = os.path.join(img_path, '%d_%d' % (data_dict[b'labels'][j], 10000*(i-1)+j)) + '.png'
            cv2.imwrite(img_name, img)

        print(data_name + ' is done')

    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_path = os.path.join(loc_2, str(test_dict[b'labels'][m]))
        dir_check(img_path, print_flag=False, empty_flag=False)
        img_name = os.path.join(img_path, '%d_%d' % (test_dict[b'labels'][m], m)) + '.png'

        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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


def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    offset += struct.calcsize(fmt_header)
    labels = []

    fmt_label = '>B'
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return labels


def export_img(exp_dir, img_ubyte, lable_ubyte):
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(lable_ubyte)

    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(exp_dir, str(labels[i]))
        dir_check(img_dir, print_flag=False)
        img_file = os.path.join(img_dir, str(i) + '.png')
        imarr = images[i]
        cv2.imwrite(img_file, imarr)


def parser_mnist_data(input_dir, output_dir):
    train_dir = output_dir[0]
    train_img_ubyte = os.path.join(input_dir, 'train-images-idx3-ubyte')
    train_label_ubyte = os.path.join(input_dir, 'train-labels-idx1-ubyte')
    export_img(train_dir, train_img_ubyte, train_label_ubyte)

    test_dir = output_dir[1]
    test_img_ubyte = os.path.join(input_dir, 't10k-images-idx3-ubyte')
    test_label_ubyte = os.path.join(input_dir, 't10k-labels-idx1-ubyte')
    export_img(test_dir, test_img_ubyte, test_label_ubyte)


if __name__ == '__main__':

    print('Please input the dataset name, you can choose Fashion_MNIST or CIFAR10：', end='')
    dataset = input()

    start = datetime.datetime.now()
    if dataset == 'CIFAR10':
        input_dir = 'Dataset\\CIFAR10'
        output_dir = ['train', 'test']
        # Empty folders
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        # Generate images
        cifar10_img(input_dir, output_dir)
    elif dataset == 'Fashion_MNIST':
        input_dir = 'Dataset\\Fashion_MNIST'
        output_dir = ['train', 'test']
        # Empty folders
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        # Generate images
        parser_mnist_data(input_dir, output_dir)