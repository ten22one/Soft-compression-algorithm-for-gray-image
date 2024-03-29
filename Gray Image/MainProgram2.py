"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import PreProcess
import CodeProcessing
import Encode
import Decode
import os
import numpy as np
import datetime
import ast

if __name__ == '__main__':
    # parameters
    input_dir = 'test'  # input folder
    output_dir = ['test_encode', 'test_decode']  # output folder
    frequency_dir = 'frequency'  # frequency folder
    codebook_dir = 'codebook'  # codebook folder
    pixel_size = 256  # the range of pixel value
    # parameters
    layer_shape = 4
    shape_height = [1, 28]  # shape height range
    shape_width = [1, 28]  # shape width range
    rough_height, rough_width = (2, 2)
    # Compression ratio and error rate
    compress_rate = np.zeros(10)  # compression ratio
    error_rate = np.ones(10)  # error rate
    start = datetime.datetime.now()  # start time
    # Clean or crate folders
    PreProcess.dir_check(output_dir[0], empty_flag=True)
    PreProcess.dir_check(output_dir[1], empty_flag=True)
    # frequency merge
    frequency_detail = {}
    frequency_rough = {}
    frequency_shape = {}
    for f in os.listdir(frequency_dir):
        for g in os.listdir(os.path.join(frequency_dir, f)):
            frequency_part = os.path.join(frequency_dir, f, g)  # Path name of the folder to be searched
            with open(frequency_part, 'r') as w:  # read the file
                frequency_mid = w.read()  # read
                frequency_mid = ast.literal_eval(frequency_mid)  # convert it into dictionary
            #  merge
            if g == 'frequency_detail.txt':
                for key in list(frequency_mid.keys()):
                    if key in frequency_detail.keys():
                        new_value = frequency_detail[key] + frequency_mid[key]  # add the number
                        frequency_detail[key] = new_value  # create the key
                    else:
                        frequency_detail[key] = 1
            elif g == 'frequency_rough.txt':
                for key in list(frequency_mid.keys()):
                    if key in frequency_rough.keys():
                        new_value = frequency_rough[key] + frequency_mid[key]  # add the number
                        frequency_rough[key] = new_value  # create the key
                    else:
                        frequency_rough[key] = 1
            elif g == 'frequency_shape.txt':
                for key in list(frequency_mid.keys()):
                    if key in frequency_shape.keys():
                        new_value = frequency_shape[key] + frequency_mid[key]  # add the number
                        frequency_shape[key] = new_value  # create the key
                    else:
                        frequency_shape[key] = 1
    # sort by frequency
    frequency_detail = dict(sorted(frequency_detail.items(), key=lambda item: item[1], reverse=True))
    frequency_rough = dict(sorted(frequency_rough.items(), key=lambda item: item[1], reverse=True))
    frequency_shape = dict(sorted(frequency_shape.items(), key=lambda item: item[1], reverse=True))
    # write the file
    frequency_merge_path = os.path.join(frequency_dir, 'merge')
    PreProcess.dir_check(frequency_merge_path, empty_flag=True)
    frequency_detail_path = os.path.join(frequency_merge_path, 'frequency_detail') + '.txt'
    frequency_rough_path = os.path.join(frequency_merge_path, 'frequency_rough') + '.txt'
    frequency_shape_path = os.path.join(frequency_merge_path, 'frequency_shape') + '.txt'
    with open(frequency_detail_path, 'w') as f:  # write the file
        f.write(str(frequency_detail))
    print('Merge result has been saved as %s' % frequency_detail_path)
    with open(frequency_rough_path, 'w') as f:  # write the file
        f.write(str(frequency_rough))
    print('Merge result has been saved as %s' % frequency_rough_path)
    with open(frequency_shape_path, 'w') as f:  # write the file
        f.write(str(frequency_shape))
    print('Merge result has been saved as %s' % frequency_shape_path)
    # Generate the codebook
    codebook_merge_dir = os.path.join(codebook_dir, 'merge')
    PreProcess.dir_check(codebook_merge_dir, empty_flag=True)
    CodeProcessing.CodeProcessing(frequency_merge_path, codebook_merge_dir)
    # Encode
    for g in os.listdir(input_dir):
        encode_dir = os.path.join(input_dir, g)  # process dir
        test_encode_dir = os.path.join(output_dir[0], g)  # output dir
        print('\n')
        print("*" * 200)  # split line
        print('Folder %s is encoding:' % encode_dir)
        rate = Encode.Encode(codebook_merge_dir, encode_dir, test_encode_dir, pixel_size, layer_shape, rough_height,
                             rough_width, start)  # encode
        compress_rate[int(g)] = rate  # compression ratio

    # Decode
    for g in os.listdir(os.path.join(output_dir[0])):
        decode_dir = os.path.join(output_dir[0], g)  # process dir
        test_decode_dir = os.path.join(output_dir[1], g)  # output dir
        original_img_dir = os.path.join(input_dir, g)  # original image dir
        print('\n')
        print("*" * 200)  # split line
        print('Folder %s is decoding:' % decode_dir)
        rate = Decode.Decode(codebook_merge_dir, decode_dir, test_decode_dir, original_img_dir, start)  # decode
        error_rate[int(g)] = rate  # error rate

    # Save compression ratio
    with open('compress_rate.txt', 'w') as g:
        g.write(str(compress_rate))
    # Save error rate
    with open('error_rate.txt', 'w') as g:
        g.write(str(error_rate))
