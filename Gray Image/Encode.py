"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import cv2
import os
import PreProcess
import numpy as np
import ast
import math
import IF
from ShapeFinding import predictive2


def string_to_bytearray(binary_string):
    """
    Convert the binary string to bytearray in every 8 bits
    :param binary_string: Input binary string
    :return: byte
    """
    # Get the last few digits that are inexhaustible by 8
    remainder_num = len(binary_string) % 8
    # Get the last few digits that are inexhaustibly divided by 8
    if remainder_num == 0:
        remainder = '00000000'
        binary_string = binary_string + format(remainder_num, 'b').zfill(8) + remainder
    else:
        remainder = binary_string[-remainder_num:].zfill(8)
        # New string: the penultimate is the number
        binary_string = binary_string[:-remainder_num] + format(remainder_num, 'b').zfill(8) + remainder
    data_bytearray = bytearray(int(binary_string[x:x + 8], 2) for x in range(0, len(binary_string), 8))
    return data_bytearray


def capacity(num):
    """
    calculate the capacity
    """
    p = num / (28 * 28)
    Hp = IF.entropy([p, 1 - p])
    cc = Hp / (1 - p)
    return cc


def Golomb(m, n):
    """
    Golomb coding for locations
    """
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
    """
    Golomb coding for locations
    """
    bit_need_total = []
    for m in range(1, int(math.pow(2, num))):
        bit_need = 0
        for value in input_list:
            encode = Golomb(m, value)
            bit_need = bit_need + len(encode)
        bit_need_total.append(bit_need)
    min_index = bit_need_total.index(min(bit_need_total)) + 1
    return min_index


def shape_encode(img_sub, height_sub, width_sub, book):
    """
    Encode the input picture
    :param img_sub: input image
    :param height_sub: height
    :param width_sub: width
    :param book: codebook
    :return:
    """
    value = {}  # Encoded value
    show = {}  # Used to visualize
    img_flag = np.zeros((height_sub, width_sub))  # Flag of whether a pixel of the picture has been encoded
    for p in list(book.keys()):
        kernel = np.array(p, np.float32)  # Convert it into a matrix
        p_height, p_width = kernel.shape  # kernel size
        dst = cv2.filter2D(img_sub, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)  # Convolve the kernel
        # Determine which positions can be coded
        can_encode_location = np.argwhere(dst == np.sum(np.power(kernel, 2)))
        for cel in can_encode_location:
            if img_flag[cel[0]][cel[1]] == 1:
                continue
            if img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width].shape != kernel.shape:
                continue
            if (img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] != kernel).any():
                continue
            can_encode_flag = img_flag[cel[0]: cel[0] + p_height,
                              cel[1]: cel[1] + p_width] == 0  # Determine whether these locations have been coded
            if can_encode_flag.all():
                try:
                    value[(cel[0], cel[1])] = book[p]  # Save the encoded value
                    show[(cel[0], cel[1])] = p  # Value for visual display
                    # Mark the position that has been coded
                    img_flag[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] = np.ones((p_height, p_width))
                except IndexError:
                    pass
    return value, show


def shape_to_binary(code_value, height_sub, width_sub):
    """
    Convert the encoded dictionary value of the shape set to binary
    :param code_value: Entered code value
    :param height_sub: height
    :param width_sub: width
    :return: binary value
    """
    binary = ''  # The binary value to be returned
    bit_height = len(format(height_sub, 'b'))  # Get the number of digits of height
    bit_width = len(format(width_sub, 'b'))  # Get the number of digits of width
    locations = list(code_value.keys())  # Get all position values
    values = list(code_value.values())  # Get all coded values
    # Convert the two-dimensional position value into one-dimensional
    locations_operate = locations[:]
    for i in range(len(locations_operate)):
        locations_operate[i] = locations_operate[i][0] * width_sub + locations_operate[i][1]
    # Used to save the position value except the first one
    locations_rest = locations_operate[1:]
    # Obtain the difference of the one-dimensional position value
    locations_difference = []
    for i in range(len(locations_rest)):
        locations_difference.append(locations_rest[i] - locations_operate[i])
    # Get the number of bits that need to encode the position difference
    try:
        Golomb_m = Golomb_m_search(locations_difference[:], 10)
    except ValueError:
        Golomb_m = 0
    # Add the number of bits required for the position difference to the total binary
    binary = binary + format(Golomb_m, 'b').zfill(10)
    # Get the position value of the final version
    for i in range(len(locations)):
        if i != 0:
            locations[i] = locations_difference[i - 1]
    for i in range(len(locations)):
        if i == 0:
            # Convert the position of height and width into a string of corresponding digits and add it to the total string
            binary = binary + format(locations[i][0], 'b').zfill(bit_height) + \
                     format(locations[i][1], 'b').zfill(bit_width)
            binary = binary + values[i]  # Add the encoded value to the string
        else:
            # The position difference is converted into a string of corresponding digits and added to the total string
            location_value = Golomb(Golomb_m, locations[i])
            binary = binary + location_value
            binary = binary + values[i]  # Add the encoded value to the string

    return binary, locations_difference[1:]


def rough_to_binary(img_sub, height_sub, width_sub, book, layer_start, rough_height, rough_width):
    """
    Convert the value of the rough layer to binary
    :param img_sub: input image
    :param height_sub: height
    :param width_sub: width
    :param book: codebook
    :return: Binary value, coded flag
    """
    # Header information
    bit_height = len(format(height_sub, 'b'))  # Get the number of digits of height
    bit_width = len(format(width_sub, 'b'))  # Get the number of digits of width
    binary = format(bit_height, 'b').zfill(4) + format(bit_width, 'b').zfill(
        4)  # Convert the digits of height and width to binary and add to the total string
    # Add layer_start to the binary
    binary = binary + format(layer_start, 'b').zfill(3)
    binary = binary + format(rough_height, 'b').zfill(3) + format(rough_width, 'b').zfill(3)
    # Add the number of bits required for the height and width of the picture to the total binary
    binary = binary + format(height_sub, 'b').zfill(bit_height) + format(width_sub, 'b').zfill(bit_width)
    img_flag = np.zeros((height_sub, width_sub))  # The sign of whether a certain pixel of the picture has been encoded
    for i in range(0, len(img_sub), rough_height):
        for j in range(0, len(img_sub[0]), rough_width):
            if i + rough_height <= len(img_sub) and j + rough_width <= len(img_sub[0]):
                # Get the number of bits required for rough encoding
                key = img_sub[i:i + rough_height, j:j + rough_width]
                key = tuple(map(tuple, key))
                mid_value = book[key]  # Find the corresponding codeword
                binary = binary + mid_value  # connect
                img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
    return binary, img_flag


def detail_to_binary(img_sub, flag, book, height_sub, width_sub):
    """
    Used to encode the pixel value of the detail layer
    :param img_sub: input image
    :param flag: encoding flag
    :param book: codebook
    :param height_sub: height
    :param width_sub: width
    :return: encoded value
    """
    binary = ''
    for i in range(height_sub):
        for j in range(width_sub):
            if flag[i][j] == 0:
                sample = img_sub[i:i + 1, j:j + 1]  # Get the sampled value there
                sample = tuple(map(tuple, sample))  # Convert to tuple
                value = book[sample]  # Get the coded value
                binary = binary + value  # Add to the total binary file
    return binary


# Encode the images
def Encode(codebook_dir, input_dir, output_dir, pixel_size, layer_start, rough_height, rough_width, start):
    # read the codebook
    with open(os.path.join(codebook_dir, 'codebook_detail.txt'), 'r') as f:
        codebook_detail = f.read()  # read
        codebook_detail = ast.literal_eval(codebook_detail)  # convert into dictionary
    with open(os.path.join(codebook_dir, 'codebook_rough.txt'), 'r') as f:
        codebook_rough = f.read()  # read
        codebook_rough = ast.literal_eval(codebook_rough)  # convert into dictionary
    with open(os.path.join(codebook_dir, 'codebook_shape.txt'), 'r') as f:
        codebook_shape = f.read()  # read
        codebook_shape = ast.literal_eval(codebook_shape)  # convert into dictionary
    encode_num = 1  # encoding number
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    folder_first, folder_num = os.path.split(output_dir)
    location_dir = os.path.join('location', str(folder_num))
    PreProcess.dir_check(location_dir, empty_flag=True)
    compress_rate = []  # Record the compression rate of each picture
    for f in os.listdir(input_dir):
        img_path = os.path.join(input_dir, f)  # image file path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read the image
        height = len(img)  # height
        width = len(img[0])  # width
        # Encoding for the shape layer
        img_predictive = predictive2(img)
        img_shape = img_predictive // math.pow(2, layer_start)
        [shape_value, shape_show] = shape_encode(img_shape, height, width, codebook_shape)
        # sort by location
        shape_value = dict(sorted(shape_value.items(), key=lambda item: (item[0][0], item[0][1])))
        binary_shape_value, location = shape_to_binary(shape_value, height, width)  # Convert to binary value
        # Get the coding value of the rough layer
        img_rough = img_predictive % math.pow(2, layer_start)
        [binary_rough_value, encode_flag] = rough_to_binary(img_rough, height, width, codebook_rough, layer_start,
                                                            rough_height, rough_width)
        # Get the code value of the detail layer
        img_detail = img_predictive % math.pow(2, layer_start)
        binary_detail_value = detail_to_binary(img_detail, encode_flag, codebook_detail, height, width)
        # binary value
        binary_value = binary_rough_value + binary_detail_value + binary_shape_value  # # Connect
        binary_bytearray = string_to_bytearray(binary_value)
        output_path = os.path.join(output_dir, f[0:f.rfind('.png')]) + '.wist'  # output file path
        with open(output_path, 'wb') as g:  # write
            g.write(binary_bytearray)
        original_pixel = height * width * len(format(pixel_size - 1, 'b'))  # The number of bits required for the original picture
        final_pixel = len(binary_value)  # The number of bits required after re-encoding
        compress_rate.append(original_pixel / final_pixel)
        print(
            '\rSaving image %d results, it needs %d bits firstly, now it needs %d with soft compression algorithm. '
            'The average compression ratio is %0.2f, minimum is %0.3f, maximum is %0.3f, variance is '
            '%0.5f' %
            (encode_num, original_pixel, final_pixel,
             np.mean(np.array(compress_rate)), min(compress_rate), max(compress_rate), np.var(compress_rate)),
            end='')
        encode_num = encode_num + 1
    return np.mean(compress_rate)
