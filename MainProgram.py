"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
MainProgram.py - Get the cross results by using soft compression algorithm for gray image
"""

import PreProcess
import ShapeFinding
import CodeProcessing
import Encode
import Decode
import os
import numpy as np
import datetime

if __name__ == '__main__':
    # Parameters
    input_dir = ['train', 'test']
    output_dir = ['test_encode', 'test_decode']
    frequency_dir = 'frequency'
    codebook_dir = 'codebook'
    pixel_size = 256
    layer_shape = 4
    shape_height = [1, 2]
    shape_width = [1, 2]
    rough_height, rough_width = (1, 1)
    # Compression rate and error rate
    compress_rate = np.zeros((10, 10))
    error_rate = np.ones((10, 10))

    start = datetime.datetime.now()
    # Clean or crate folders
    PreProcess.dir_check(frequency_dir, empty_flag=True)
    PreProcess.dir_check(codebook_dir, empty_flag=True)
    PreProcess.dir_check(output_dir[0], empty_flag=True)
    PreProcess.dir_check(output_dir[1], empty_flag=True)
    # Shape searching
    for f in os.listdir(input_dir[0]):
        process_dir = os.path.join(input_dir[0], f)
        after_process_dir = os.path.join(frequency_dir, f)
        print("*" * 150)
        print('\nFolder %s is searching:' % process_dir)
        ShapeFinding.ShapeFinding(process_dir, after_process_dir, layer_shape, shape_height, shape_width, rough_height,
                                  rough_width, start)
    # Codebook processing
    for f in os.listdir(frequency_dir):
        process_frequency = os.path.join(frequency_dir, f)
        after_process_dir = os.path.join(codebook_dir, f)
        print("*" * 150)
        print('\nFolder %s is processing：' % process_frequency)
        CodeProcessing.CodeProcessing(process_frequency, after_process_dir)

    # Encode
    for f in os.listdir(codebook_dir):
        codebook_use_dir = os.path.join(codebook_dir, f)
        for g in os.listdir(input_dir[1]):
            encode_dir = os.path.join(input_dir[1], g)
            test_encode_dir = os.path.join(output_dir[0], f, g)
            print('\n')
            print("*" * 230)
            print('%s-----%s is encoding:' % (codebook_use_dir, encode_dir))
            rate = Encode.Encode(codebook_use_dir, encode_dir, test_encode_dir, pixel_size, layer_shape, rough_height,
                                 rough_width, start)
            compress_rate[int(f)][int(g)] = rate

    # Decode
    for f in os.listdir(output_dir[0]):
        codebook_use_dir = os.path.join(codebook_dir, f)
        for g in os.listdir(os.path.join(output_dir[0], f)):
            decode_dir = os.path.join(output_dir[0], f, g)
            test_decode_dir = os.path.join(output_dir[1], f, g)
            original_img_dir = os.path.join(input_dir[1], g)
            print('\n')
            print("*" * 230)
            print('Folder %s is decoding:' % decode_dir)
            rate = Decode.Decode(codebook_use_dir, decode_dir, test_decode_dir, original_img_dir, start)
            error_rate[int(f)][int(g)] = rate  # 得到差错率

    # Save compression ratio
    with open('compress_rate.txt', 'w') as g:
        g.write(str(compress_rate))
    # Save error rate
    with open('error_rate.txt', 'w') as g:
        g.write(str(error_rate))
