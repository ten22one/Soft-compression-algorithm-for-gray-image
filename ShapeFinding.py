"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import cv2
import os
import datetime
import numpy as np
import math
import PreProcess
import itertools


def entropy(pd):
    """
    Calculate the entropy of a random variable
    pd: input probability distribution
    """
    Entropy = 0
    for p in pd:
        if p != 0:
            Entropy = Entropy - p * math.log2(p)
    return Entropy


def new_represent2(img):
    """
    Get the predictive error of an image
    :param img: input image
    :return: predictive error
    """
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_return = np.zeros(img.shape, np.float32)  # save the output data
    # predictive
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ic, B, dh, dv = GAP(img, i, j)
            ew = get(img, i, j - 1) - GAP(img, i, j - 1)[0]
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
            try:
                N[C] = N[C] + 1
            except KeyError:
                N[C] = 1
            try:
                S[C] = S[C] + math.floor(img[i][j] - Itilde)
            except KeyError:
                S[C] = math.floor(img[i][j] - Itilde)
            if N[C] == 128:
                N[C] = int(N[C] / 2)
                S[C] = S[C] / 2
            #  Get the value in normal range
            img_return[i][j] = Itilde
            if img_return[i][j] > 255:
                img_return[i][j] = 255
            elif img_return[i][j] < 0:
                img_return[i][j] = 0
            #  Get the integer value
            img_return[i][j] = math.floor(img[i][j] - img_return[i][j])
    # Mapping from negative to positive
    img_return = negative_to_positive(img_return)
    return img_return


def new_represent(img):
    """
    Get the predictive error of an image
    :param img: input image
    :return: prediction error
    """
    img_original = img  # input image
    img_fill = np.zeros((img_original.shape[0] + 1, img_original.shape[1] + 1))
    img_fill[0:1, :] = np.zeros((1, img_fill.shape[1]))
    img_fill[1:img_fill.shape[0], 1:img_fill.shape[1]] = img_original
    img_fill[1:img_fill.shape[0], 0:1] = img_fill[0:img_fill.shape[0] - 1, 1:2]
    img_return = np.zeros(img_fill.shape, np.int16)  # Return value
    # Prediction
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
    # Calculate the prediction error
    img_return = img_original - img_return[1:img_return.shape[0], 1:img_return.shape[1]]
    # Mapping from negative to positive
    img_return = negative_to_positive(img_return)
    return img_return


def get(im, i, j):
    """
    For an image, returning the integer value on (i,j). The aim is to prevent index errors.
    :param im: input image
    :param i: height
    :param j: width
    :return: If it is in the image, return its value; Otherwise, return zero.
    """
    if 0 <= i < im.shape[0] and 0 <= j < im.shape[1]:
        return int(im[i, j])
    return 0


def GAP(im, i, j):
    """
    For an image, get the prediction error on (i,j). Prediction value is based on Gradient-Adjusted Prediction.
    :param im: Input image
    :param i: height
    :param j: width
    :return: prediction value
    """
    # Context
    In = get(im, i - 1, j)
    Iw = get(im, i, j - 1)
    Ine = get(im, i - 1, j + 1)
    Inw = get(im, i - 1, j - 1)
    Inn = get(im, i - 2, j)
    Iww = get(im, i, j - 2)
    Inne = get(im, i - 2, j + 1)

    # Process the boundary
    if i == 0 and j > 0:
        In = get(im, i, j - 1)
        Inn = get(im, i, j - 1)
    if i == 1 and j > 0:
        Inn = get(im, i - 1, j - 1)
    if j == 0 and i > 0:
        Iw = get(im, i - 1, j)
        Iww = get(im, i - 1, j)
    if j == 1 and i > 0:
        Iww = get(im, i - 1, j - 1)
    # Input
    dh = abs(Iw - Iww) + abs(In - Inw) + abs(In - Ine)
    dv = abs(Iw - Inw) + abs(In - Inn) + abs(Ine - Inne)
    # GAP
    if dv - dh > 80:
        ic = Iw
    elif dv - dh < -80:
        ic = In
    else:
        ic = (Iw + In) / 2 + (Ine - Inw) / 4
        if dv - dh > 32:
            ic = (ic + Iw) / 2
        elif dv - dh > 8:
            ic = (3 * ic + Iw) / 4
        elif dv - dh < -32:
            ic = (ic + In) / 2
        elif dv - dh < -8:
            ic = (3 * ic + In) / 4
    temp = list(map(lambda x: int(x < ic), [(2 * Iw) - Iww, (2 * In) - Inn, Iww, Inn, Ine, Inw, Iw, In]))
    B = temp[0] << 7 | temp[1] << 6 | temp[2] << 5 | temp[3] << 4 | temp[4] << 3 | temp[5] << 2 | temp[6] << 1 | temp[7]
    return ic, B, dh, dv


def BGRtoYUV(b, g, r):
    """
    Color space conversion from bgr to yuv
    """
    y = (r + 2 * g + b) / 4
    y = np.floor(y)
    u = r - g
    v = b - g
    return y, u, v


def YUVtoBGR(y, u, v):
    """
    color space conversion from yuv to bgr
    """
    inverse_g = y - np.floor((u + v) / 4)
    inverse_r = u + inverse_g
    inverse_b = v + inverse_g
    return inverse_b, inverse_g, inverse_r


def negative_to_positive(img):
    """
    Mapping from negative to positive
    """
    img_new = np.zeros(img.shape)  # return image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                img_new[i][j] = 2 * img[i][j]
            elif img[i][j] < 0:
                img_new[i][j] = -2 * img[i][j] - 1
    return img_new


def mode_select():
    """
    Judge to adopt which prediction method
    """
    sample = os.listdir('train')[0]  # input image
    img_path = os.path.join("train", sample)  # image path
    img = cv2.imread(img_path)  # read the image
    b, g, r = cv2.split(img)  # split the image
    img, u, v = BGRtoYUV(b, g, r)  # color space conversion from bgr to yuv
    img_i = new_represent(img) // 16
    img_ii = new_represent2(img) // 16
    sta_i = np.zeros((32,))
    sta_ii = np.zeros((32,))
    # Calculate the entropy
    for detail in range(32):
        number_i = np.sum(img_i == detail)
        sta_i[detail] = number_i
        number_ii = np.sum(img_ii == detail)
        sta_ii[detail] = number_ii
    sta_i = sta_i / sum(sta_i)
    sta_ii = sta_ii / sum(sta_ii)
    entropy_i = entropy(sta_i)
    entropy_ii = entropy(sta_ii)
    print(entropy_i, entropy_ii)
    if entropy_i > entropy_ii:
        return 8, 5, (1, 1), (1, 2), 'ii'
    else:
        return 4, 4, (1, 2), (1, 2), 'i'


class DetailSearch:
    """
    Detail search
    """
    def __init__(self):
        self.input_dir = 'train'  # input folder
        self.output_dir = 'frequency'  # output folder
        self.pixel_size = 256  # the range of pixel value
        self.layer_start = 4  # shape layer start
        self.detail_set = {}
        self.component = 'b'
        self.mode = 'i'

    def main(self):
        """
        Main search work
        :return: None
        """
        img_path_total = os.listdir(self.input_dir)  # image file path
        search_num = 1  # search number
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)  # image path
            img = cv2.imread(img_path)  # read the image
            img = img.astype(np.int16)
            b, g, r = cv2.split(img)  # split the image
            y, u, v = BGRtoYUV(b, g, r)  # mapping from bgr to yuv
            if self.component == 'y':
                img = y
            elif self.component == 'u':
                img = u
            elif self.component == 'v':
                img = v
            # Get the prediction value
            if self.mode == 'ii':
                img = new_represent2(img)
            else:
                img = new_represent(img)
            # Get the shape layer
            img = img % math.pow(2, self.layer_start)
            self.search_shape(img)  # search each shape's frequency
            print('\rSearching for image %d' % search_num, end='')
            search_num = search_num + 1
        # The smallest shape
        for key in range(int(math.pow(2, self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))
            if key in self.detail_set.keys():
                pass
            else:
                self.detail_set[key] = 1  # create
        # output frequency file
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'detail' + '_' + self.component + '.txt'  # output name
        # save the file
        with open(output_name, 'w') as f:  # write the file
            f.write(str(self.detail_set))
        print('\nSearch is complete，it has been saved as %s' % output_name)

    def search_shape(self, img):
        """
        Get the frequency of each shape
        :param img: input image
        :return: None
        """
        for detail in range(int(math.pow(2, self.layer_start))):
            number = np.sum(img == detail)
            detail = tuple(map(tuple, np.array([[detail]])))  # convert to tuple
            self.renew_set(detail, number)  # renew the set

    def renew_set(self, detail, number):
        """
        Renew the set of shapes
        :param detail: input
        :param number: frequency
        :return:
        """
        if number != 0:
            if detail in self.detail_set.keys():
                self.detail_set[detail] = self.detail_set[detail] + number  # add the number
            else:
                self.detail_set[detail] = number  # create the tuple


class RoughSearch:
    """
    Rough search
    """
    def __init__(self):
        self.input_dir = 'train'  # input folder
        self.output_dir = 'frequency'  # output folder
        self.pixel_size = 256  # the range of pixel value
        self.layer_start = 4  # shape layer start
        self.batch_size = 1
        self.rough_set = {}
        self.rough_height = 2
        self.rough_width = 2
        self.component = 'b'
        self.mode = 'i'

    def main(self):
        """
        Main search work
        """
        img_path_total = os.listdir(self.input_dir)  # image file path
        search_num = 1  # search number
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)  # image path
            img = cv2.imread(img_path)  # read the image
            img = img.astype(np.int16)
            b, g, r = cv2.split(img)  # split the image
            y, u, v = BGRtoYUV(b, g, r)  # mapping from bgr to yuv
            if self.component == 'y':
                img = y
            elif self.component == 'u':
                img = u
            elif self.component == 'v':
                img = v
            # get the prediction value
            if self.mode == 'ii':
                img = new_represent2(img)
            else:
                img = new_represent(img)
            # layer separation
            img = img % math.pow(2, self.layer_start)
            img = img.astype(np.int16)
            # searching
            self.search_shape(img)  # Get the frequency
            print('\rSearching for image %d' % search_num, end='')
            search_num = search_num + 1
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'rough' + '_' + self.component + '.txt'  # output file
        # The smallest shape
        key_range = range(int(math.pow(2, self.layer_start)))
        for key in itertools.product(key_range, repeat=self.rough_height * self.rough_width):
            key = np.array(key)
            key = np.reshape(key, (self.rough_height, self.rough_width))
            key = tuple(map(tuple, key))  # convert into tuple
            if key in self.rough_set.keys():
                pass
            else:
                self.rough_set[key] = 1  # create the key
        # Sort by frequency
        self.rough_set = dict(sorted(self.rough_set.items(), key=lambda item: item[1], reverse=True))
        # save the file
        with open(output_name, 'w') as f:  # write the file
            f.write(str(self.rough_set))
        print('\nSearch is complete，it has been saved as %s' % output_name)

    def search_shape(self, img):
        """
        Get the frequency of each shape
        :param img: input image
        :return: None
        """
        for i in range(0, len(img), self.rough_height):
            for j in range(0, len(img[0]), self.rough_width):
                if i + self.rough_height <= len(img) and j + self.rough_width <= len(img[0]):
                    value = img[i:i + self.rough_height, j:j + self.rough_width]  # value
                    value = tuple(map(tuple, value))  # convert into tuple
                    self.renew_set(value)  # renew the set

    def renew_set(self, sample):
        """
        Renew the set of shapes
        :param sample: input
        :return: frequency
        """
        if sample in self.rough_set.keys():
            new_value = self.rough_set[sample] + 1  # add the number
            self.rough_set[sample] = new_value  # create the tuple
        else:
            self.rough_set[sample] = 1


class ShapeSearch:
    """
    Shape Search
    """
    def __init__(self):
        self.input_dir = 'train'  # input folder
        self.output_dir = 'frequency'  # output folder
        self.batch_size = 1  # batch size
        self.shape_height = [1, 4]  # shape height
        self.shape_width = [1, 4]  # shape width
        self.pixel_size = 256  # the range of pixel value
        self.layer_start = 4  # layer interface
        self.layer_end = int(math.log2(self.pixel_size)) + 1  # layer end
        self.shape_set = {}  # shape set
        self.degree = 0.1
        self.component = 'b'
        self.mode = 'i'

    def main(self):
        img_path_total = os.listdir(self.input_dir)  # image file path
        round_total = math.ceil(len(img_path_total) / self.batch_size)  # search round
        round_num = 0  # round num
        for i in range(round_total):
            # the address of image
            start_num = self.batch_size * round_num  # start location
            end_num = self.batch_size * (round_num + 1)  # end location
            # image path batch
            try:
                img_path_batch = img_path_total[start_num: end_num]
            except IndexError:
                img_path_batch = img_path_total[start_num:]
            # Renew the set
            for f in img_path_batch:
                img_path = os.path.join(self.input_dir, f)  # image path
                img = cv2.imread(img_path)  # read the image
                img = img.astype(np.int16)
                b, g, r = cv2.split(img)  # split the image
                y, u, v = BGRtoYUV(b, g, r)  # mapping from bgr to yuv
                if self.component == 'y':
                    img = y
                elif self.component == 'u':
                    img = u
                elif self.component == 'v':
                    img = v
                # Get the prediction value
                if self.mode == 'ii':
                    img = new_represent2(img)
                else:
                    img = new_represent(img)
                # Get the shape layer
                img = img // math.pow(2, self.layer_start)
                img = img.astype(np.int16)
                self.search_shape(img)  # search each shape's frequency
            del_num = self.shape_compress(degree2=round_num + 1, degree3=self.batch_size)  # delete shapes
            print('\rSearching round %d' % (round_num + 1), end='')
            round_num = round_num + 1
        # The smallest shape
        if self.component == 'y':
            increase_range = int(math.pow(2, self.layer_end - self.layer_start))
        else:
            increase_range = int(math.pow(2, self.layer_end + 1 - self.layer_start))
        for key in range(1, increase_range):
            key = tuple(map(tuple, np.array([[key]])))  # convert into tuple
            if key in self.shape_set.keys():
                pass
            else:
                self.shape_set[key] = 1  # create
        # sort according to frequency
        shape_set = dict(sorted(self.shape_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'shape' + '_' + self.component + '.txt'  # output file
        # save the file
        with open(output_name, 'w') as f:  # write
            f.write(str(shape_set))
        print('\nSearch is complete，it has been saved as %s' % output_name)

    def search_shape(self, img):
        """
        Get the frequency of each shape
        :param img: input image
        :return: None
        """
        # candidate location
        lo_total = np.argwhere(img != 0)
        for lo in lo_total:
            (i, j) = lo
            for u in range(self.shape_height[0], self.shape_height[1] + 1):
                for v in range(self.shape_width[0], self.shape_width[1] + 1):
                    self.get_sample(img, [i, j, u, v])  # get the sample

    def get_sample(self, img, index):
        """
        Get the sample
        :param img: input image
        :param index: location index
        :return: None
        """
        if (index[0] + index[2] <= len(img)) and (index[1] + index[3] <= len(img[0])):
            sample = img[index[0]:index[0] + index[2], index[1]:index[1] + index[3]]  # sample value
            sample = tuple(map(tuple, sample))  # convert into tuple
            self.renew_set(sample)  # renew the set

    def renew_set(self, sample):
        """
        Renew the set of shapes
        :param sample: input sample
        :return: None
        """
        sample_judge = np.array(sample) == 0
        if (np.sum(sample_judge, axis=0) <= sample_judge.shape[0] / 2).all() \
                and (np.sum(sample_judge, axis=1) <= sample_judge.shape[1] / 2).all():  # the condition of shapes
            if sample in self.shape_set.keys():
                new_value = self.shape_set[sample] + 1  # add the number
                self.shape_set[sample] = new_value  # renew
            else:
                self.shape_set[sample] = 1  # create the sample

    def shape_compress(self, degree2, degree3):
        """
        delete shapes
       :param degree2: delete parameter
       :param degree3: delete parameter
       :return: None
       """
        num = 0
        for key in list(self.shape_set.keys()):
            if self.shape_set[key] <= self.degree * degree2 * degree3:
                del self.shape_set[key]  # delete the shape
                num = num + 1
        return num


# Find the set of shapes
def ShapeFinding(dataset):
    input_dir = 'train'  # input folder
    output_dir = 'frequency'  # output folder
    # layer interface
    layer_shape = 4
    batch_size = 1  # batch size
    shape_degree = 0.5  # delete degree
    shape_height, shape_width = ([1, 4], [1, 4])  # shape height and width
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    print("*" * 150, '\n')
    # Mode
    if dataset == ('drive' or 'PH2'):
        layer_shape_y, layer_shape_uv = 4, 4
        rough_y, rough_uv = (1, 2), (1, 2)
        mode = 'i'
    else:
        layer_shape_y, layer_shape_uv, rough_y, rough_uv, mode = mode_select()
    component_total = ['y', 'u', 'v']  # three components
    for component in component_total:
        if component == 'y':  # y component
            layer_shape = layer_shape_y
            rough_height, rough_width = rough_y
        elif component == 'u':  # u component
            layer_shape = layer_shape_uv
            rough_height, rough_width = rough_uv
        elif component == 'v':  # v component
            layer_shape = layer_shape_uv
            rough_height, rough_width = rough_uv
        # Detail search
        FS = DetailSearch()
        FS.layer_start, FS.component, FS.mode = layer_shape, component, mode
        FS.main()
        # print("*" * 150, '\n')
        # Rough search
        SS = RoughSearch()
        SS.layer_start, SS.component, SS.rough_height, SS.rough_width, SS.mode = layer_shape, component, rough_height, rough_width, mode
        SS.main()
        # print("*" * 150, '\n')
        # Shape search
        TS = ShapeSearch()
        TS.shape_height, TS.shape_width, TS.layer_start, TS.batch_size, TS.degree, TS.component, TS.mode \
            = shape_height, shape_width, layer_shape, batch_size, shape_degree, component, mode
        TS.main()
        # print("*" * 150, '\n')
    return mode
