"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import cv2
import os
import PreProcess
import numpy as np
import datetime
import ast
import math
from ShapeFinding import GAP
from ShapeFinding import get


def bytearray_to_string(bytearray_path):
    """
    Convert the entered address into a string
    :param bytearray_path: input address
    :return: string
    """
    # get the size
    bytearray_size = os.path.getsize(bytearray_path)
    data_total = ''
    with open(bytearray_path, 'rb') as g:
        # Get the previous value
        for i in range(bytearray_size - 2):
            data = g.read(1)  # Output one byte at a time
            data = ord(bytes(data))
            data = format(data, 'b').zfill(8)
            data_total += data
        # Get the number
        last_num = g.read(1)  # Output one byte at a time
        last_num = ord(bytes(last_num))
        # Get the final value
        if last_num != 0:
            last_data = g.read(1)
            last_data = ord(bytes(last_data))
            last_data = format(last_data, 'b').zfill(last_num)
            data_total += last_data
    # Output string
    return data_total


def anti_Golomb(m, binary):
    """
    the pseudo process of Golomb coding
    """
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
    """
    Golomb coding
    """
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
    """
    Decode the input encoded file into image
    :param tt: Encoded binary file
    :return: image
    """

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

    # header information
    bit_height = int(tt[0:4], 2)  # Get the number of digits of height
    bit_width = int(tt[4:8], 2)  # Get the number of digits of the width
    layer_start = int(tt[8:11], 2)  # Get the starting layer
    rough_height = int(tt[11:14], 2)
    rough_width = int(tt[14:17], 2)
    height = int(tt[17:17 + bit_height], 2)
    width = int(tt[17 + bit_height: 17 + bit_height + bit_width], 2)
    img_rough = np.zeros((height, width))  # Create restored picture
    img_flag = np.zeros((height, width))  # Create the flag bit of the restored picture
    tt = tt[17 + bit_height + bit_width:]  # Remove the beginning information of tt
    # rough
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
                        # 给标志位置位
                        img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
                        break
                    num = num + 1
                point_num = point_num + num
    tt = tt[point_num:]
    # detail
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
    # shape
    img_shape = np.zeros((height, width))  # Create restored picture
    # First shape decoding
    Golomb_m = int(tt[0:10], 2)  # location difference
    tt = tt[10:]
    if len(tt) != 0 and Golomb_m != 1:
        first_height = int(tt[0:bit_height],
                           2)  # Determine the height in the starting cell and convert it to an integer type
        first_width = int(tt[bit_height:bit_height + bit_width],
                          2)  # Determine the width in the starting cell and convert it to an integer type
        tt = tt[bit_height + bit_width:]  # Delete the high position value and the wide position value
        value, tt = decode_search(tt, decode_shape)
        shape_height = len(value)
        shape_width = len(value[0])
        img_shape[first_height: first_height + shape_height, first_width: first_width + shape_width] = np.array(value)
        last_location = first_height * width + first_width
    # The remaining shape decoding
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
        # Assign a value to the point at the corresponding position
        shape_height = len(value)  # shape height
        shape_width = len(value[0])  # shape width
        img_shape[location_height: location_height + shape_height,
        location_width: location_width + shape_width] = np.array(value)  # get the shape
    img_restore = restore(img_shape, img_rough, layer_start)
    return img_restore


def decode_search(input_tt, book):
    """
    Search the input string, starting from 0, until one can find the corresponding string in the decoding table
    :param input_tt: input string
    :param book: codebook
    :return:
    """
    for i in range(1, len(input_tt) + 1):
        encode_value = input_tt[0:i]
        if encode_value in book.keys():
            decode_value = book[encode_value]
            return decode_value, input_tt[i:]


def restore(img1, img2, layer):
    """
    restore the image
    """
    img_positive = img1 * math.pow(2, layer) + img2  # positive value
    img_negative = np.zeros(img_positive.shape)  # negative value
    for i in range(len(img_negative)):
        for j in range(len(img_negative[0])):
            x = img_positive[i][j]
            if x != 0:
                if x % 2 == 0:
                    img_negative[i][j] = x / 2
                elif x % 2 == 1:
                    img_negative[i][j] = - (x + 1) / 2
    img_final = anti_predicting(img_negative)  # anti prediction
    # generate an image
    return img_final


def anti_predicting(img):
    """
    anti prediction
    """
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_bp = np.zeros(img.shape, np.float32)  # return value
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
    """
    To calculate the root mean square error of two images.
    :param input1: input image
    :param input2: input image
    :return: fidelity rate
    """
    fidelity_rate = 0
    difference = input1 - input2  # calculate the difference
    for i in range(len(difference)):
        for j in range(len(difference[0])):
            fidelity_rate = fidelity_rate + pow(difference[i][j], 2)  # Get the sum of the squares of the difference
    fidelity_rate = fidelity_rate / (len(difference) * len(difference[0]))  # Divide by the image area
    fidelity_rate = pow(fidelity_rate, 0.5)
    return fidelity_rate


# Decoder
def Decode(codebook_dir, input_dir, output_dir, original_img_dir, start):
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    # read the codebook
    with open(os.path.join(codebook_dir, 'codebook_rough.txt'), 'r') as f:
        codebook_rough = f.read()  # read the file
        codebook_rough = ast.literal_eval(codebook_rough)  # convert into dictionary
    with open(os.path.join(codebook_dir, 'codebook_detail.txt'), 'r') as f:
        codebook_detail = f.read()  # read the file
        codebook_detail = ast.literal_eval(codebook_detail)  # convert into dictionary
    with open(os.path.join(codebook_dir, 'codebook_shape.txt'), 'r') as f:
        codebook_shape = f.read()  # read the file
        codebook_shape = ast.literal_eval(codebook_shape)  # convert into dictionary
    decode_rough = {v: k for k, v in codebook_rough.items()}  # Flip the codebook to generate the codebook needed for decoding
    decode_detail = {v: k for k, v in codebook_detail.items()}
    decode_shape = {v: k for k, v in codebook_shape.items()}
    max_rough_length = 0
    for key in list(decode_rough.keys()):
        middle_length = len(key)
        if middle_length > max_rough_length:
            max_rough_length = middle_length
    error_rate_total = []  # Used to save a list of total fidelity
    num = 1  # counting
    # Read the encoded file and convert it into an image
    for f in os.listdir(input_dir):
        tt_path = os.path.join(input_dir, f)
        if os.path.splitext(tt_path)[1] == '.wist':  # The directory contains .wist files
            img_encode = bytearray_to_string(tt_path)
            img = decode(img_encode, decode_rough, decode_detail, decode_shape)  # Decode the encoded file
            img_original_path = os.path.join(original_img_dir, f[0:f.rfind('.wist')]) + '.png'  # The path of the uncompressed picture
            output_path = os.path.join(output_dir, f[0:f.rfind('.wist')]) + '.png'  # the output file name
            cv2.imwrite(output_path, img)  # Save the decoded image
            img_original = cv2.imread(img_original_path, cv2.IMREAD_GRAYSCALE)  # Read in the uncompressed picture
            error_rate = fidelity(img_original, img)  # Gain fidelity
            error_rate_total.append(error_rate)  # Add to total fidelity
            end = datetime.datetime.now()
            print('\rSaving image %d, root mean square error is %0.2f, the average root mean square error is %0.2f'
                  % (num, error_rate, np.mean(error_rate_total)), end='')
            num = num + 1
    return np.mean(error_rate_total)
