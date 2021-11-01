"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import PreProcess
import ShapeFinding
import CodeProcessing
import Encode
import Decode
import os
import numpy as np
import datetime
from multiprocessing import Process
if __name__ == '__main__':
    # Parameters
    input_dir = ['train', 'test']  # input folder
    output_dir = ['test_encode', 'test_decode']  # output folder
    frequency_dir = 'frequency'  # frequency folder
    codebook_dir = 'codebook'  # codebook folder
    pixel_size = 256  # the range of pixel value
    # Parameters
    layer_shape = 4
    shape_height = [1, 28]  # shape height range
    shape_width = [1, 28]  # shape width range
    rough_height, rough_width = (2, 2)
    # Compression ratio and error rate
    compress_rate = np.zeros((10, 10))  # compression ratio
    error_rate = np.ones((10, 10))  # error rate
    start = datetime.datetime.now()  # start time
    # Clean or crate folders
    PreProcess.dir_check(frequency_dir, empty_flag=True)
    PreProcess.dir_check(codebook_dir, empty_flag=True)
    PreProcess.dir_check(output_dir[0], empty_flag=True)
    PreProcess.dir_check(output_dir[1], empty_flag=True)
    # Shape searching
    mp = []
    for f in os.listdir(input_dir[0]):
        process_dir = os.path.join(input_dir[0], f)  # process dir
        after_process_dir = os.path.join(frequency_dir, f)  # output dir
        print("*" * 100)  # split line
        print('\nFolder %s is searching:' % process_dir)
        ShapeFinding.ShapeFinding(process_dir, after_process_dir, layer_shape, shape_height, shape_width,
                                    rough_height, rough_width, start)  # Find the set of shapes
    # Codebook processing
    for f in os.listdir(frequency_dir):
        process_frequency = os.path.join(frequency_dir, f)  # process dir
        after_process_dir = os.path.join(codebook_dir, f)  # output dir
        print("*" * 150)  # split line
        print('\nFolder %s is processing：' % process_frequency)
        CodeProcessing.CodeProcessing(process_frequency, after_process_dir)  # Generate the codebook

    # Encode
    for f in os.listdir(codebook_dir):
        codebook_use_dir = os.path.join(codebook_dir, f)  # codebook path
        for g in os.listdir(input_dir[1]):
            encode_dir = os.path.join(input_dir[1], g)  # process dir
            test_encode_dir = os.path.join(output_dir[0], f, g)  # output dir
            print('\n')
            print("*" * 200)  # split line
            print('%s-----%s is encoding:' % (codebook_use_dir, encode_dir))
            rate = Encode.Encode(codebook_use_dir, encode_dir, test_encode_dir, pixel_size, layer_shape, rough_height, rough_width, start)  # encode
            compress_rate[int(f)][int(g)] = rate  # compression ratio

    # Decode
    for f in os.listdir(output_dir[0]):
        codebook_use_dir = os.path.join(codebook_dir, f)  # codebook path
        for g in os.listdir(os.path.join(output_dir[0], f)):
            decode_dir = os.path.join(output_dir[0], f, g)  # process dir
            test_decode_dir = os.path.join(output_dir[1], f, g)  # output dir
            original_img_dir = os.path.join(input_dir[1], g)  # original image folder
            print('\n')
            print("*" * 200)  # split line
            print('Folder %s is decoding:' % decode_dir)
            rate = Decode.Decode(codebook_use_dir, decode_dir, test_decode_dir, original_img_dir, start)  # decode
            error_rate[int(f)][int(g)] = rate  # error rate

    # Save compression ratio
    with open('compress_rate.txt', 'w') as g:
        g.write(str(compress_rate))
    # Save error rate
    with open('error_rate.txt', 'w') as g:
        g.write(str(error_rate))
