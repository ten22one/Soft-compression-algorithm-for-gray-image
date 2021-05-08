"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import cv2
import os
import PreProcess_s
import numpy as np
import datetime
import ast
import math
from ShapeFinding_s import negative_to_positive
from ShapeFinding_s import predictive2


def string_to_bytearray(binary_string):
    remainder_num = len(binary_string) % 8
    if remainder_num == 0:
        remainder = '00000000'
        binary_string = binary_string + format(remainder_num, 'b').zfill(8) + remainder
    else:
        remainder = binary_string[-remainder_num:].zfill(8)
        binary_string = binary_string[:-remainder_num] + format(remainder_num, 'b').zfill(8) + remainder
    data_bytearray = bytearray(int(binary_string[x:x + 8], 2) for x in range(0, len(binary_string), 8))
    return data_bytearray


def Golomb(m, n):
    n = n - 1
    q = math.floor(n / m) * '1' + '0'

    k = math.ceil(math.log2(m))
    c = int(math.pow(2, k)) - m
    r = n % m
    if 0 <= r < c:
        rr = format(r, 'b').zfill(k - 1)
    else:
        rr = format(r + c, 'b').zfill(k)
    value = q + rr
    return value


def Golomb_m_search(input_list, num):
    bit_need_total = []
    for m in range(1, int(math.pow(2, num))):
        bit_need = 0
        for value in input_list:
            encode = Golomb(m, value)
            bit_need = bit_need + len(encode)
        bit_need_total.append(bit_need)
    min_index = bit_need_total.index(min(bit_need_total)) + 1
    return min_index


def predictive(img):
    img_original = img
    img_fill = np.zeros((img_original.shape[0] + 1, img_original.shape[1] + 1))

    img_fill[0:1, :] = np.zeros((1, img_fill.shape[1]))
    img_fill[1:img_fill.shape[0], 1:img_fill.shape[1]] = img_original
    img_fill[1:img_fill.shape[0], 0:1] = img_fill[0:img_fill.shape[0] - 1, 1:2]

    img_return = np.zeros(img_fill.shape, np.int16)

    for i in range(1, img_fill.shape[0]):
        for j in range(1, img_fill.shape[1]):
            a = int(img_fill[i][j - 1])
            b = int(img_fill[i - 1][j])
            c = int(img_fill[i - 1][j - 1])
            if c >= max(a, b):
                img_return[i][j] = min(a, b)
            elif c < min(a, b):
                img_return[i][j] = max(a, b)
            else:
                img_return[i][j] = a + b - c
    img_return = img_original - img_return[1:img_return.shape[0], 1:img_return.shape[1]]
    img_return = negative_to_positive(img_return)
    return img_return


def shape_encode(img_sub, height_sub, width_sub, book):
    value = {}
    show = {}
    img_flag = np.zeros((height_sub, width_sub))
    for p in list(book.keys()):
        kernel = np.array(p, np.float32)
        p_height, p_width = kernel.shape
        dst = cv2.filter2D(img_sub, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        can_encode_location = np.argwhere(dst == np.sum(np.power(kernel, 2)))
        for cel in can_encode_location:
            if img_flag[cel[0]][cel[1]] == 1:
                continue
            if img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width].shape != kernel.shape:
                continue
            if (img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] != kernel).any():
                continue
            can_encode_flag = img_flag[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] == 0
            if can_encode_flag.all():
                try:
                    value[(cel[0], cel[1])] = book[p]
                    show[(cel[0], cel[1])] = p
                    img_flag[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] = np.ones((p_height, p_width))
                except IndexError:
                    pass
    return value, show


def shape_to_binary(code_value, height_sub, width_sub):
    binary = ''
    bit_height = len(format(height_sub, 'b'))
    bit_width = len(format(width_sub, 'b'))
    locations = list(code_value.keys())
    values = list(code_value.values())

    locations_operate = locations[:]
    for i in range(len(locations_operate)):
        locations_operate[i] = locations_operate[i][0] * width_sub + locations_operate[i][1]
    locations_rest = locations_operate[1:]
    locations_difference = []
    for i in range(len(locations_rest)):
        locations_difference.append(locations_rest[i] - locations_operate[i])

    try:
        Golomb_m = Golomb_m_search(locations_difference[:], 10)
    except ValueError:
        Golomb_m = 0
    binary = binary + format(Golomb_m, 'b').zfill(10)
    for i in range(len(locations)):
        if i != 0:
            locations[i] = locations_difference[i - 1]
    for i in range(len(locations)):
        if i == 0:
            binary = binary + format(locations[i][0], 'b').zfill(bit_height) + \
                     format(locations[i][1], 'b').zfill(bit_width)
            binary = binary + values[i]
        else:
            location_value = Golomb(Golomb_m, locations[i])
            binary = binary + location_value
            binary = binary + values[i]
    return binary, locations_difference[1:]


def rough_to_binary(img_sub, height_sub, width_sub, book, layer_start, rough_height, rough_width):
    bit_height = len(format(height_sub, 'b'))
    bit_width = len(format(width_sub, 'b'))
    binary = format(bit_height, 'b').zfill(4) + format(bit_width, 'b').zfill(4)
    binary = binary + format(layer_start, 'b').zfill(3)
    binary = binary + format(rough_height, 'b').zfill(3) + format(rough_width, 'b').zfill(3)
    binary = binary + format(height_sub, 'b').zfill(bit_height) + format(width_sub, 'b').zfill(bit_width)
    img_flag = np.zeros((height_sub, width_sub))
    for i in range(0, len(img_sub), rough_height):
        for j in range(0, len(img_sub[0]), rough_width):
            if i + rough_height <= len(img_sub) and j + rough_width <= len(img_sub[0]):
                key = img_sub[i:i + rough_height, j:j + rough_width]
                key = tuple(map(tuple, key))
                mid_value = book[key]
                binary = binary + mid_value
                img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
    return binary, img_flag


def detail_to_binary(img_sub, flag, book, height_sub, width_sub):
    binary = ''
    for i in range(height_sub):
        for j in range(width_sub):
            if flag[i][j] == 0:
                sample = img_sub[i:i + 1, j:j + 1]
                sample = tuple(map(tuple, sample))
                value = book[sample]
                binary = binary + value
    return binary


def Encode(codebook_dir, input_dir, output_dir, pixel_size, layer_start, rough_height, rough_width, start):
    with open(os.path.join(codebook_dir, 'codebook_detail.txt'), 'r') as f:
        codebook_detail = f.read()
        codebook_detail = ast.literal_eval(codebook_detail)
    with open(os.path.join(codebook_dir, 'codebook_rough.txt'), 'r') as f:
        codebook_rough = f.read()
        codebook_rough = ast.literal_eval(codebook_rough)
    with open(os.path.join(codebook_dir, 'codebook_shape.txt'), 'r') as f:
        codebook_shape = f.read()
        codebook_shape = ast.literal_eval(codebook_shape)

    encode_num = 1
    PreProcess_s.dir_check(output_dir, empty_flag=True)
    compress_rate = []
    rough_rate = []
    shape_rate = []
    for f in os.listdir(input_dir):
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height = len(img)
        width = len(img[0])

        img_predictive = predictive2(img)
        img_shape = img_predictive // math.pow(2, layer_start)
        [shape_value, shape_show] = shape_encode(img_shape, height, width, codebook_shape)
        shape_value = dict(sorted(shape_value.items(), key=lambda item: (item[0][0], item[0][1])))
        binary_shape_value, location = shape_to_binary(shape_value, height, width)
        img_rough = img_predictive % math.pow(2, layer_start)
        [binary_rough_value, encode_flag] = rough_to_binary(img_rough, height, width, codebook_rough, layer_start,
                                                            rough_height, rough_width)
        img_detail = img_predictive % math.pow(2, layer_start)
        binary_detail_value = detail_to_binary(img_detail, encode_flag, codebook_detail, height, width)
        binary_value = binary_rough_value + binary_detail_value + binary_shape_value
        binary_bytearray = string_to_bytearray(binary_value)
        output_path = os.path.join(output_dir, f[0:f.rfind('.png')]) + '.wist'
        with open(output_path, 'wb') as g:
            g.write(binary_bytearray)
        original_pixel = height * width * len(format(pixel_size - 1, 'b'))
        final_pixel = len(binary_value)
        end = datetime.datetime.now()
        compress_rate.append(original_pixel / final_pixel)
        rough_rate.append(len(binary_rough_value))
        shape_rate.append(len(binary_shape_value))
        print(
            '\rSaving image %d results, it needs %d bits firstly, now needs %d by using soft compression algorithm. '
            'Program has run %s. Average compression ratio is %0.2f, minimum is %0.3f, maximum is %0.3f, variance is '
            '%0.5f' %
            (encode_num, original_pixel, final_pixel, end - start,
             np.mean(np.array(compress_rate)), min(compress_rate), max(compress_rate), np.var(compress_rate)),
            end='')
        encode_num = encode_num + 1
    return np.mean(compress_rate)
