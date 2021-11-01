"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import cv2
import os
import datetime
import numpy as np
import PreProcess
import itertools
import math


def predictive2(img):
    """
    Get the predictive error of an image
    :param img: input image
    :return: predictive error
    """
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_return = np.zeros(img.shape, np.float32)  # save the output data
    # prediction
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


class DetailSearch:
    """
    Detail search
    """

    def __init__(self):
        self.input_dir = 'train'  # input folder
        self.output_dir = 'frequency'  # output folder
        self.pixel_size = 256  # the range of pixel value
        self.layer_start = 4  # shape layer start
        self.start = datetime.datetime.now()
        self.detail_set = {}

    def main(self):
        """
        Main search work
        :return: None
        """
        img_path_total = os.listdir(self.input_dir)  # image file path
        search_num = 1  # search number
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)  # image path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read the image
            img = predictive2(img)  # Get the prediction value
            img = img % math.pow(2, self.layer_start)  # layer separation
            self.search_shape(img)  # search the shape
            print('\rSearching for image %d' % search_num, end='')
            search_num = search_num + 1
        # The smallest shape
        for key in range(int(math.pow(2, self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))  # convert into tuple
            if key in self.detail_set.keys():
                pass
            else:
                self.detail_set[key] = 1  # create the tuple
        self.detail_set = dict(sorted(self.detail_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'detail' + '.txt'  # output file path
        # write and save the file
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
            detail = tuple(map(tuple, np.array([[detail]])))  # convert into tuple
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
        self.layer_start = 3  # shape layer start
        self.batch_size = 1
        self.start = datetime.datetime.now()
        self.rough_set = {}
        self.rough_height = 2
        self.rough_width = 2

    def main(self):
        """
        Main search work
        """
        img_path_total = os.listdir(self.input_dir)  # image file path
        search_num = 1  # search number
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)  # image path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read the image
            img = predictive2(img)  # split the image
            img = img % math.pow(2, self.layer_start)
            img = img.astype(np.int16)
            self.search_shape(img)  # Get the frequency
            print('\rSearching for image %d' % search_num, end='')
            search_num = search_num + 1
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'rough' + '.txt'  # output file name
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
        self.start = datetime.datetime.now()
        self.shape_set = {}  # shape set
        self.degree = 0.1

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
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read the image
                img = predictive2(img)  # prediction
                img = img // math.pow(2, self.layer_start)  # layer separation
                img = img.astype(np.int16)
                self.search_shape(img)  # search each shape's frequency
            del_num = self.shape_compress(degree2=round_num + 1, degree3=self.batch_size)  # delete shapes
            print('\rSearching round %d' % (round_num + 1), end='')
            round_num = round_num + 1
        # The smallest shape
        for key in range(1, int(math.pow(2, self.layer_end - self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))  # convert into tuple
            if key in self.shape_set.keys():
                pass
            else:
                self.shape_set[key] = 1  # create
        # sort according to frequency
        shape_set = dict(sorted(self.shape_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'shape' + '.txt'  # output file
        # save the file
        with open(output_name, 'w') as f:  # write the file
            f.write(str(shape_set))
        print('\nSearch is complete，it has been saved as %s' % output_name)

    def search_shape(self, img):
        """
        Get the frequency of each shape
        :param img: input image
        :return: None
        """
        height = len(img)  # height
        width = len(img[0])  # width
        for i in range(height):
            for j in range(width):
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
                self.shape_set[sample] = new_value  # renew the dictionary
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
def ShapeFinding(input_dir, output_dir, layer_shape, shape_height, shape_width, rough_height, rough_width, start):
    # parameter
    batch_size = 10  # batch size
    shape_degree = 0.1
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    # Detail
    FS = DetailSearch()
    FS.input_dir, FS.output_dir, FS.layer_start, FS.start = input_dir, output_dir, layer_shape, start  # Pass parameters
    FS.main()
    # Rough
    SS = RoughSearch()
    SS.input_dir, SS.output_dir, SS.layer_start, SS.rough_height, SS.rough_width, SS.start \
        = input_dir, output_dir, layer_shape, rough_height, rough_width, start  # Pass parameters
    SS.main()
    # shape
    TS = ShapeSearch()
    TS.input_dir, TS.output_dir, TS.shape_height, TS.shape_width, TS.layer_start, TS.batch_size, TS.degree, TS.start \
        = input_dir, output_dir, shape_height, shape_width, layer_shape, batch_size, shape_degree, start  # Pass parameters
    TS.main()
