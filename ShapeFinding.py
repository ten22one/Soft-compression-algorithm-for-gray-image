"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
ShapeFinding.py - Find the frequency set
"""

import cv2
import os
import datetime
import numpy as np
import PreProcess
import itertools
import math


def predictive2(img):
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_return = np.zeros(img.shape, np.float32)

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

            img_return[i][j] = Itilde
            if img_return[i][j] > 255:
                img_return[i][j] = 255
            elif img_return[i][j] < 0:
                img_return[i][j] = 0

            img_return[i][j] = math.floor(img[i][j] - img_return[i][j])
            if ed < 0:
                img_return[i][j] = -img_return[i][j]

    img_return = negative_to_positive(img_return)
    return img_return


def get(im, i, j):
    if 0 <= i < im.shape[0] and 0 <= j < im.shape[1]:
        return int(im[i, j])
    return 0


def GAP(im, i, j):
    In = get(im, i - 1, j)
    Iw = get(im, i, j - 1)
    Ine = get(im, i - 1, j + 1)
    Inw = get(im, i - 1, j - 1)
    Inn = get(im, i - 2, j)
    Iww = get(im, i, j - 2)
    Inne = get(im, i - 2, j + 1)

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
    img_new = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                img_new[i][j] = 2 * img[i][j]
            elif img[i][j] < 0:
                img_new[i][j] = -2 * img[i][j] - 1
    return img_new


class DetailSearch:

    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.pixel_size = 256
        self.layer_start = 4
        self.start = datetime.datetime.now()
        self.detail_set = {}

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        search_num = 1
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = predictive2(img)
            img = img % math.pow(2, self.layer_start)

            self.search_shape(img)
            end = datetime.datetime.now()
            print('\rDetail layer: Number %d is searching, the number of codewords is %d, program has run %s'
                  % (search_num, len(self.detail_set), end - self.start), end='')
            search_num = search_num + 1
        for key in range(int(math.pow(2, self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))
            if key in self.detail_set.keys():
                pass
            else:
                self.detail_set[key] = 1
        self.detail_set = dict(sorted(self.detail_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'detail' + '.txt'
        with open(output_name, 'w') as f:
            f.write(str(self.detail_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def search_shape(self, img):
        for detail in range(int(math.pow(2, self.layer_start))):
            number = np.sum(img == detail)
            detail = tuple(map(tuple, np.array([[detail]])))
            self.renew_set(detail, number)

    def renew_set(self, detail, number):
        if number != 0:
            if detail in self.detail_set.keys():
                self.detail_set[detail] = self.detail_set[detail] + number
            else:
                self.detail_set[detail] = number


class RoughSearch:

    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.pixel_size = 256
        self.layer_start = 4
        self.batch_size = 1
        self.start = datetime.datetime.now()
        self.rough_set = {}
        self.rough_height = 2
        self.rough_width = 2

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        search_num = 1
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = predictive2(img)
            img = img % math.pow(2, self.layer_start)
            img = img.astype(np.int16)
            self.search_shape(img)
            end = datetime.datetime.now()
            print('\rRough Layer: Number %d is searching, the number of codewords is %d, program has run %s'
                  % (search_num, len(self.rough_set), end - self.start), end='')
            search_num = search_num + 1
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'rough' + '.txt'

        key_range = range(int(math.pow(2, self.layer_start)))
        for key in itertools.product(key_range, repeat=self.rough_height * self.rough_width):
            key = np.array(key)
            key = np.reshape(key, (self.rough_height, self.rough_width))
            key = tuple(map(tuple, key))
            if key in self.rough_set.keys():
                pass
            else:
                self.rough_set[key] = 1
        self.rough_set = dict(sorted(self.rough_set.items(), key=lambda item: item[1], reverse=True))
        with open(output_name, 'w') as f:
            f.write(str(self.rough_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def search_shape(self, img):
        for i in range(0, len(img), self.rough_height):
            for j in range(0, len(img[0]), self.rough_width):
                if i + self.rough_height <= len(img) and j + self.rough_width <= len(img[0]):
                    value = img[i:i + self.rough_height, j:j + self.rough_width]
                    value = tuple(map(tuple, value))
                    self.renew_set(value)

    def renew_set(self, sample):
        if sample in self.rough_set.keys():
            new_value = self.rough_set[sample] + 1
            self.rough_set[sample] = new_value
        else:
            self.rough_set[sample] = 1


class ShapeSearch:

    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.batch_size = 1
        self.shape_height = [1, 4]
        self.shape_width = [1, 4]
        self.pixel_size = 256
        self.layer_start = 4
        self.layer_end = int(math.log2(self.pixel_size)) + 1
        self.start = datetime.datetime.now()
        self.shape_set = {}
        self.degree = 0.1

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        round_total = math.ceil(len(img_path_total) / self.batch_size)
        round_num = 0
        for i in range(round_total):
            start_num = self.batch_size * round_num
            end_num = self.batch_size * (round_num + 1)
            try:
                img_path_batch = img_path_total[start_num: end_num]
            except IndexError:
                img_path_batch = img_path_total[start_num:]

            for f in img_path_batch:
                img_path = os.path.join(self.input_dir, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = predictive2(img)
                img = img // math.pow(2, self.layer_start)
                img = img.astype(np.int16)
                self.search_shape(img)

            del_num = self.shape_compress(degree2=round_num + 1, degree3=self.batch_size)
            end = datetime.datetime.now()
            print('\rShape Layer: Number %d is searching, the number of codewords is %d, delete num is %d, '
                  'program has run %s '
                  % (round_num + 1, len(self.shape_set), del_num, end - self.start), end='')
            round_num = round_num + 1
        for key in range(1, int(math.pow(2, self.layer_end - self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))
            if key in self.shape_set.keys():
                pass
            else:
                self.shape_set[key] = 1
        shape_set = dict(sorted(self.shape_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir, 'frequency') + '_' + 'shape' + '.txt'
        with open(output_name, 'w') as f:
            f.write(str(shape_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def search_shape(self, img):
        height = len(img)
        width = len(img[0])
        for i in range(height):
            for j in range(width):
                for u in range(self.shape_height[0], self.shape_height[1] + 1):
                    for v in range(self.shape_width[0], self.shape_width[1] + 1):
                        self.get_sample(img, [i, j, u, v])

    def get_sample(self, img, index):
        if (index[0] + index[2] <= len(img)) and (index[1] + index[3] <= len(img[0])):
            sample = img[index[0]:index[0] + index[2], index[1]:index[1] + index[3]]
            sample = tuple(map(tuple, sample))
            self.renew_set(sample)

    def renew_set(self, sample):
        sample_judge = np.array(sample) == 0
        if (np.sum(sample_judge, axis=0) <= sample_judge.shape[0] / 2).all() \
                and (np.sum(sample_judge, axis=1) <= sample_judge.shape[1] / 2).all():
            if sample in self.shape_set.keys():
                new_value = self.shape_set[sample] + 1
                self.shape_set[sample] = new_value
            else:
                self.shape_set[sample] = 1

    def shape_compress(self, degree2, degree3):
        num = 0
        for key in list(self.shape_set.keys()):
            if self.shape_set[key] <= self.degree * degree2 * degree3:
                del self.shape_set[key]
                num = num + 1
        return num


def ShapeFinding(input_dir, output_dir, layer_shape, shape_height, shape_width, rough_height, rough_width, start):
    # Given parameter
    batch_size = 10
    shape_degree = 0.1
    PreProcess.dir_check(output_dir, empty_flag=True)

    # Detail
    FS = DetailSearch()
    FS.input_dir, FS.output_dir, FS.layer_start, FS.start = input_dir, output_dir, layer_shape, start
    FS.main()
    # Rough
    SS = RoughSearch()
    SS.input_dir, SS.output_dir, SS.layer_start, SS.rough_height, SS.rough_width, SS.start \
        = input_dir, output_dir, layer_shape, rough_height, rough_width, start
    SS.main()
    # Shape
    TS = ShapeSearch()
    TS.input_dir, TS.output_dir, TS.shape_height, TS.shape_width, TS.layer_start, TS.batch_size, TS.degree, TS.start \
        = input_dir, output_dir, shape_height, shape_width, layer_shape, batch_size, shape_degree, start
    TS.main()
