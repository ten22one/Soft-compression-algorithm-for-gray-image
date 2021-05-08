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
from ShapeFinding_s import GAP
from ShapeFinding_s import get


def bytearray_to_string(bytearray_path):
    bytearray_size = os.path.getsize(bytearray_path)
    data_total = ''
    with open(bytearray_path, 'rb') as g:
        for i in range(bytearray_size - 2):
            data = g.read(1)
            data = ord(bytes(data))
            data = format(data, 'b').zfill(8)
            data_total += data
        last_num = g.read(1)
        last_num = ord(bytes(last_num))
        if last_num != 0:
            last_data = g.read(1)
            last_data = ord(bytes(last_data))
            last_data = format(last_data, 'b').zfill(last_num)
            data_total += last_data
    return data_total


def anti_Golomb(m, binary):
    index = binary.find('0')
    q = index
    k = math.ceil(math.log2(m))
    c = int(math.pow(2, k)) - m

    rr = int(binary[index + 1:], 2)
    rr_length = len(binary[index + 1:])
    if rr_length == k - 1:
        r = rr
    elif rr_length == k:
        r = rr - c

    value = q * m + r
    value = value + 1
    return value


def Golomb_search(input, m):
    q_index = input.find('0')
    q = q_index

    k = math.ceil(math.log2(m))
    c = int(math.pow(2, k)) - m

    try:
        rr1 = int(input[q_index + 1: q_index + 1 + k - 1], 2)
        n1 = q * m + rr1
        r1 = n1 % m

        if 0 <= r1 < c:
            value = input[0:q_index + 1 + k - 1]
            output_value = input[q_index + 1 + k - 1:]
        else:
            value = input[0:q_index + 1 + k]
            output_value = input[q_index + 1 + k:]
    except ValueError:
        value = input[0:q_index + 1 + k]
        output_value = input[q_index + 1 + k:]

    value = anti_Golomb(m, value)
    return value, output_value


def decode(tt, decode_rough, decode_detail, decode_shape):
    def Golomb_search2(number, m):
        q_index = tt.find('0', number)
        q = q_index

        k = math.ceil(math.log2(m))
        c = int(math.pow(2, k)) - m

        try:
            rr1 = int(tt[q_index + 1: q_index + 1 + k - 1], 2)
            n1 = q * m + rr1
            r1 = n1 % m

            if 0 <= r1 < c:
                value = tt[0 + number:q_index + 1 + k - 1]
                output_value = q_index + 1 + k - 1
            else:
                value = tt[0 + number:q_index + 1 + k]
                output_value = q_index + 1 + k
        except ValueError:
            value = tt[0 + number:q_index + 1 + k]
            output_value = q_index + 1 + k
        value = anti_Golomb(m, value)
        return value, output_value

    bit_height = int(tt[0:4], 2)
    bit_width = int(tt[4:8], 2)
    layer_start = int(tt[8:11], 2)
    rough_height = int(tt[11:14], 2)
    rough_width = int(tt[14:17], 2)
    height = int(tt[17:17 + bit_height], 2)
    width = int(tt[17 + bit_height: 17 + bit_height + bit_width], 2)

    img_rough = np.zeros((height, width))
    img_flag = np.zeros((height, width))
    tt = tt[17 + bit_height + bit_width:]

    point_num = 0
    for i in range(0, len(img_rough), rough_height):
        for j in range(0, len(img_rough[0]), rough_width):
            if i + rough_height <= len(img_rough) and j + rough_width <= len(img_rough[0]):
                num = 1
                while True:
                    encode_value = tt[point_num:point_num + num]
                    if encode_value in decode_rough.keys():
                        rough_value = decode_rough[encode_value]
                        img_rough[i: i + rough_height, j: j + rough_width] = np.array(rough_value)
                        img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
                        break
                    num = num + 1
                point_num = point_num + num
    tt = tt[point_num:]

    point_num = 0
    for i in range(len(img_rough)):
        for j in range(len(img_rough[0])):
            if img_flag[i][j] == 0:
                num = 1
                while True:
                    encode_value = tt[point_num:point_num + num]
                    if encode_value in decode_detail.keys():
                        detail_value = decode_detail[encode_value]
                        break
                    num = num + 1
                point_num = point_num + num
                img_rough[i][j] = np.array(detail_value)
    tt = tt[point_num:]
    img_shape = np.zeros((height, width))
    Golomb_m = int(tt[0:10], 2)
    tt = tt[10:]
    if len(tt) != 0 and Golomb_m != 1:
        first_height = int(tt[0:bit_height], 2)
        first_width = int(tt[bit_height:bit_height + bit_width], 2)
        tt = tt[bit_height + bit_width:]
        value, tt = decode_search(tt, decode_shape)
        shape_height = len(value)
        shape_width = len(value[0])
        img_shape[first_height: first_height + shape_height, first_width: first_width + shape_width] = np.array(value)
        last_location = first_height * width + first_width
    point_num = 0
    while True:
        if point_num == len(tt):
            break
        location_relative, num2 = Golomb_search2(point_num, Golomb_m)
        point_num = num2
        if location_relative == 0:
            break
        location = last_location + location_relative
        last_location = location
        location_height = location // width
        location_width = location % width

        num = 1
        while True:
            encode_value = tt[point_num:point_num + num]
            if encode_value in decode_shape.keys():
                value = decode_shape[encode_value]
                break
            num = num + 1
        point_num = point_num + num
        shape_height = len(value)
        shape_width = len(value[0])
        img_shape[location_height: location_height + shape_height,
        location_width: location_width + shape_width] = np.array(value)
    img_restore = restore(img_shape, img_rough, layer_start)
    return img_restore


def decode_search(input_tt, book):
    for i in range(1, len(input_tt) + 1):
        encode_value = input_tt[0:i]
        if encode_value in book.keys():
            decode_value = book[encode_value]
            return decode_value, input_tt[i:]


def restore(img1, img2, layer):
    img_positive = img1 * math.pow(2, layer) + img2
    img_negative = np.zeros(img_positive.shape)
    for i in range(len(img_negative)):
        for j in range(len(img_negative[0])):
            x = img_positive[i][j]
            if x != 0:
                if x % 2 == 0:
                    img_negative[i][j] = x / 2
                elif x % 2 == 1:
                    img_negative[i][j] = - (x + 1) / 2
    img_final = anti_predicting(img_negative)
    return img_final


def anti_predicting(img):
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_bp = np.zeros(img.shape, np.float32)
    img_bp[0][0] = img[0][0]
    img_ap = np.zeros(img.shape, np.float32)
    for i in range(img_bp.shape[0]):
        for j in range(img_bp.shape[1]):
            ic, B, dh, dv = GAP(img_bp, i, j)
            ew = get(img_bp, i, j - 1) - GAP(img_bp, i, j - 1)[0]
            delt = dh + dv + 2 * abs(ew)
            Qdel = -1
            k = 0
            while k < len(thre):
                if delt <= thre[k]:
                    Qdel = k
                    break
                k = k + 1
            if Qdel == -1:
                Qdel = 7
            C = B * Qdel
            try:
                ed = S[C] / N[C]
            except KeyError:
                ed = 0
            Itilde = ic + ed

            img_ap[i][j] = Itilde
            if img_ap[i][j] > 255:
                img_ap[i][j] = 255
            elif img_ap[i][j] < 0:
                img_ap[i][j] = 0
            img_bp[i][j] = math.ceil(img_ap[i][j] + img[i][j])
            try:
                N[C] = N[C] + 1
            except KeyError:
                N[C] = 1
            try:
                S[C] = S[C] + math.floor(img_bp[i][j] - Itilde)
            except KeyError:
                S[C] = math.floor(img_bp[i][j] - Itilde)
            if N[C] == 128:
                N[C] = int(N[C] / 2)
                S[C] = S[C] / 2
    return img_bp


def fidelity(input1, input2):
    fidelity_rate = 0
    difference = input1 - input2
    for i in range(len(difference)):
        for j in range(len(difference[0])):
            fidelity_rate = fidelity_rate + pow(difference[i][j], 2)
    fidelity_rate = fidelity_rate / (len(difference) * len(difference[0]))
    fidelity_rate = pow(fidelity_rate, 0.5)
    return fidelity_rate


def Decode(codebook_dir, input_dir, output_dir, original_img_dir, start):
    PreProcess_s.dir_check(output_dir, empty_flag=True)
    with open(os.path.join(codebook_dir, 'codebook_rough.txt'), 'r') as f:
        codebook_rough = f.read()
        codebook_rough = ast.literal_eval(codebook_rough)
    with open(os.path.join(codebook_dir, 'codebook_detail.txt'), 'r') as f:
        codebook_detail = f.read()
        codebook_detail = ast.literal_eval(codebook_detail)
    with open(os.path.join(codebook_dir, 'codebook_shape.txt'), 'r') as f:
        codebook_shape = f.read()
        codebook_shape = ast.literal_eval(codebook_shape)

    decode_rough = {v: k for k, v in codebook_rough.items()}
    decode_detail = {v: k for k, v in codebook_detail.items()}
    decode_shape = {v: k for k, v in codebook_shape.items()}

    max_rough_length = 0
    for key in list(decode_rough.keys()):
        middle_length = len(key)
        if middle_length > max_rough_length:
            max_rough_length = middle_length

    error_rate_total = []
    num = 1
    for f in os.listdir(input_dir):
        tt_path = os.path.join(input_dir, f)
        if os.path.splitext(tt_path)[1] == '.wist':
            img_encode = bytearray_to_string(tt_path)
            img = decode(img_encode, decode_rough, decode_detail, decode_shape)
            img_original_path = os.path.join(original_img_dir, f[0:f.rfind('.wist')]) + '.png'
            output_path = os.path.join(output_dir, f[0:f.rfind('.wist')]) + '.png'
            cv2.imwrite(output_path, img)

            img_original = cv2.imread(img_original_path, cv2.IMREAD_GRAYSCALE)
            error_rate = fidelity(img_original, img)
            error_rate_total.append(error_rate)

            end = datetime.datetime.now()
            print('\rSaving image %d, root mean square error is %0.2f, average root mean square error is %0.2f, '
                  'program has run %s '
                  % (num, error_rate, np.mean(error_rate_total), end - start), end='')
            num = num + 1
    return np.mean(error_rate_total)
