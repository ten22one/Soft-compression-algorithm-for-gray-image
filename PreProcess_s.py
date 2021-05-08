"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import os
import shutil
import cv2
import datetime
import struct
import numpy as np


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
    dataset = 'FASHION-MNIST'
    start = datetime.datetime.now()
    if dataset == 'FASHION-MNIST':
        input_dir = 'Dataset\\FASHION'
        output_dir = ['train', 'test']
        # Empty folders
        dir_check(output_dir[0], empty_flag=True)
        dir_check(output_dir[1], empty_flag=True)
        # Generate images
        parser_mnist_data(input_dir, output_dir)
