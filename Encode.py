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
import arithmeticcoding
from ShapeFinding import BGRtoYUV, new_represent, new_represent2


def header_generator(bit_num, input):
    """
    generate the header
    """
    binary_input = format(input, 'b')  # binary input
    step = bit_num - 1
    if len(binary_input) % step != 0:
        zfill_num = len(binary_input) + step - len(binary_input) % 9
        binary_input = binary_input.zfill(zfill_num)
    batch = [binary_input[i:i + step] for i in range(0, len(binary_input), step)]  # batch
    output = ''  # output
    for i in range(len(batch)):
        if i == len(batch) - 1:
            output += '1' + batch[i].zfill(step)
        else:
            output += '0' + batch[i].zfill(step)
    return output


def bytearray_to_string(bytearray_path):
    """
    Convert the entered address into string
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


def Golomb(m, n):
    """
    Golomb coding for locations
    """
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
        kernel = np.array(p, np.float32)  # Convert it to a matrix
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
    if len(locations) != 0 and Golomb_m == 1:
        Golomb_m = 2
    if len(locations) == 0:
        Golomb_m = 1
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
    binary = binary + Golomb(Golomb_m, 0)
    return binary, locations_difference[1:]


def rough_to_binary(img_sub, height_sub, width_sub, book, layer_start, rough_height, rough_width, space):
    """
    Convert the value of the rough layer to binary
    :param img_sub: input image
    :param height_sub: height
    :param width_sub: width
    :param book: codebook
    :return: Binary value, coded flag
    """
    # Header information
    binary = ''
    img_flag = np.zeros((height_sub, width_sub))  # Flag of whether a pixel of the picture has been encoded
    rough_number = int(math.pow(2, layer_start))
    rough_number = int(math.pow(rough_number, rough_height * rough_width))
    # arithmetic coding
    bitout = arithmeticcoding.BitOutputStream()
    initfreqs = arithmeticcoding.FlatFrequencyTable(rough_number)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    freqs.frequencies = [int(book[i], 2) for i in book.keys()]
    freqs.total = sum(freqs.frequencies)
    for i in range(0, len(img_sub), rough_height):
        for j in range(0, len(img_sub[0]), rough_width):
            if i + rough_height <= len(img_sub) and j + rough_width <= len(img_sub[0]):
                key_matrix = img_sub[i:i + rough_height, j:j + rough_width]
                key = 0
                for m in range(key_matrix.shape[0]):
                    for n in range(key_matrix.shape[1]):
                        key = key + int(
                            key_matrix[m][n] * math.pow(math.pow(2, layer_start), key_matrix.size - m - n - 1))
                img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
                enc.write(freqs, key)
                freqs.increment(key)
    enc.finish()
    binary_value = ''.join(enc.output.binary)
    binary_head = header_generator(10, len(binary_value))
    binary = binary_head + binary_value  # binary value
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


def space_binary(img, codebook_shape, codebook_rough, codebook_detail, layer_start, rough_height, rough_width, space,
                 mode):
    height = len(img)  # Get the height of the picture (counting from 0)
    width = len(img[0])  # Get the width of the picture (counting from 0)
    # prediction
    if mode == 'ii':
        img_predict = new_represent2(img)
    else:
        img_predict = new_represent(img)
    # Detail
    img_rough = img_predict % math.pow(2, layer_start)
    img_detail = img_rough
    img_shape = img_predict // math.pow(2, layer_start)
    # Rough
    binary_rough_value, encode_flag = rough_to_binary(img_rough, height, width, codebook_rough, layer_start,
                                                      rough_height, rough_width, space)
    binary_detail_value = detail_to_binary(img_detail, encode_flag, codebook_detail, height, width)
    # shape
    [shape_value, shape_show] = shape_encode(img_shape, height, width, codebook_shape)
    # sort
    shape_value = dict(sorted(shape_value.items(), key=lambda item: (item[0][0], item[0][1])))
    binary_shape_value, location = shape_to_binary(shape_value, height, width)  # Convert to binary value
    # Save the encoded binary file
    binary_value = binary_rough_value + binary_detail_value + binary_shape_value  # Binary connection of all layers
    return binary_value


# Encode the images
def Encode(dataset, mode):
    codebook_dir = 'codebook'  # codebook dir
    input_dir = 'test'  # input file
    output_dir = 'test_encode'  # output file
    pixel_size = 256  # pixel range
    layer_end = int(math.log2(pixel_size)) + 1  # layer end
    # prediction
    if mode == "ii":
        layer_shape_y, layer_shape_uv = 8, 5
        rough_y, rough_uv = (1, 1), (1, 2)
    elif mode == "i":
        layer_shape_y, layer_shape_uv = 4, 4
        rough_y, rough_uv = (1, 2), (1, 2)
    rough_height_y, rough_width_y = rough_y
    rough_height_uv, rough_width_uv = rough_uv
    print("*" * 150, '\n')
    # read the codebook
    for txt in os.listdir(codebook_dir):
        with open(os.path.join(codebook_dir, txt), 'r') as f:
            codebook = f.read()
            codebook = ast.literal_eval(codebook)
        if txt == 'codebook_detail_y.txt':
            codebook_detail_y = codebook
        elif txt == 'codebook_rough_y.txt':
            codebook_rough_y = codebook
        elif txt == 'codebook_shape_y.txt':
            codebook_shape_y = codebook
        elif txt == 'codebook_detail_u.txt':
            codebook_detail_u = codebook
        elif txt == 'codebook_rough_u.txt':
            codebook_rough_u = codebook
        elif txt == 'codebook_shape_u.txt':
            codebook_shape_u = codebook
        elif txt == 'codebook_detail_v.txt':
            codebook_detail_v = codebook
        elif txt == 'codebook_rough_v.txt':
            codebook_rough_v = codebook
        elif txt == 'codebook_shape_v.txt':
            codebook_shape_v = codebook
    encode_num = 1  # encode num
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    compress_rate = []  # Record the compression rate of each picture
    for f in os.listdir(input_dir):
        num = int(f[0:f.rfind('.png')])
        img_path = os.path.join(input_dir, f)  # image file path
        img = cv2.imread(img_path)  # read the image
        img = img.astype(np.int16)
        b, g, r = cv2.split(img)  # split the image
        y, u, v = BGRtoYUV(b, g, r)  # convert the image from bgr to yuv
        (height, width) = b.shape  # height and width of an image
        bit_height = len(format(height, 'b'))  # Get the number of digits of height
        bit_width = len(format(width, 'b'))  # Get the number of digits of width
        binary = format(bit_height, 'b').zfill(4) + format(bit_width, 'b').zfill(4)  # Convert the digits of height and width to binary and add to the total string
        # header information
        binary = binary + format(height, 'b').zfill(bit_height) + format(width, 'b').zfill(bit_width)
        binary = binary + format(layer_shape_y - 1, 'b').zfill(3)
        binary = binary + format(rough_height_y, 'b').zfill(3) + format(rough_width_y, 'b').zfill(3)
        binary = binary + format(layer_shape_uv - 1, 'b').zfill(3)
        binary = binary + format(rough_height_uv, 'b').zfill(3) + format(rough_width_uv, 'b').zfill(3)
        if mode == 'ii':
            binary = binary + '0'
        else:
            binary = binary + '1'
        # space encoding for y component
        binary_b = space_binary(y, codebook_shape_y, codebook_rough_y, codebook_detail_y, layer_shape_y, rough_height_y,
                                rough_width_y, 'y', mode)
        # space encoding for u component
        binary_g = space_binary(u, codebook_shape_u, codebook_rough_u, codebook_detail_u, layer_shape_uv,
                                rough_height_uv, rough_width_uv, 'u', mode)
        # space encoding for v component
        binary_r = space_binary(v, codebook_shape_v, codebook_rough_v, codebook_detail_v, layer_shape_uv,
                                rough_height_uv, rough_width_uv, 'v', mode)
        # binary value
        binary_value = binary + binary_b + binary_g + binary_r
        binary_bytearray = string_to_bytearray(binary_value)  # byte array
        output_path = os.path.join(output_dir, f[0:f.rfind('.png')]) + '.wist'  # output file name
        with open(output_path, 'wb') as g:  # write the file
            g.write(binary_bytearray)
        original_pixel = height * width * len(format(pixel_size - 1, 'b')) * 3  # Original number of bits
        final_pixel = len(binary_value)  # final number of bits
        compress_rate.append(original_pixel / final_pixel)  # compression ratio
        print(
            '\rSaving image %d results, it needs %d bits firstly, and now it needs %d bits with soft compression algorithm. '
            'The average compression ratio is %0.3f, minimum is %0.3f, maximum is %0.3f, variance is '
            '%0.5f' %
            (encode_num, original_pixel, final_pixel,
             np.mean(np.array(compress_rate)), min(compress_rate), max(compress_rate), np.var(compress_rate)),
            end='')
        encode_num = encode_num + 1
