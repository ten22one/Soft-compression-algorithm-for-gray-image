import cv2
import os
import numpy as np
import math


def predictive2(img):
    N = {}
    S = {}
    thre = [5, 15, 25, 42, 60, 85, 140]
    img_return = np.zeros(img.shape, np.float32)
    img_predict = np.zeros(img.shape, np.float32)
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

            img_predict[i][j] = math.floor(Itilde)
            if img_predict[i][j] > 255:
                img_predict[i][j] = 255
            elif img_predict[i][j] < 0:
                img_predict[i][j] = 0

            img_return[i][j] = img[i][j] - img_predict[i][j]
    img_return = negative_to_positive(img_return, img_predict)
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


def fidelity(input1, input2):
    fidelity_rate = 0
    difference = input1 - input2
    for i in range(len(difference)):
        for j in range(len(difference[0])):
            fidelity_rate = fidelity_rate + pow(difference[i][j], 2)
    fidelity_rate = fidelity_rate / (len(difference) * len(difference[0]))
    fidelity_rate = pow(fidelity_rate, 0.5)
    fidelity_rate = np.mean(fidelity_rate)
    return fidelity_rate


def negative_to_positive(error, img_predict):
    img_new = np.zeros(error.shape)
    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
            if img_predict[i][j] <= 128:
                if error[i][j] <= img_predict[i][j]:
                    if error[i][j] > 0:
                        img_new[i][j] = 2 * error[i][j]
                    elif error[i][j] < 0:
                        img_new[i][j] = -2 * error[i][j] - 1
                else:
                    img_new[i][j] = error[i][j] + img_predict[i][j]
            else:
                if (-error[i][j]) <= (255 - img_predict[i][j]):
                    if error[i][j] > 0:
                        img_new[i][j] = 2 * error[i][j]
                    elif error[i][j] < 0:
                        img_new[i][j] = -2 * error[i][j] - 1
                else:
                    img_new[i][j] = - error[i][j] + 255 - img_predict[i][j]
    return img_new


def anti_predicting(img):
    img = img.astype(np.float32)
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
            img_ap[i][j] = math.floor(Itilde)

            if img_ap[i][j] > 255:
                img_ap[i][j] = 255
            elif img_ap[i][j] < 0:
                img_ap[i][j] = 0

            if img_ap[i][j] <= 128:
                if img[i][j] <= 2 * img_ap[i][j]:
                    if img[i][j] != 0:
                        if img[i][j] % 2 == 0:
                            img[i][j] = img[i][j] / 2
                        elif img[i][j] % 2 == 1:
                            img[i][j] = -(img[i][j] + 1) / 2
                else:
                    img[i][j] = img[i][j] - img_ap[i][j]
            else:
                if img[i][j] <= 2 * (255-img_ap[i][j]):
                    if img[i][j] != 0:
                        if img[i][j] % 2 == 0:
                            img[i][j] = img[i][j] / 2
                        elif img[i][j] % 2 == 1:
                            img[i][j] = -(img[i][j] + 1) / 2
                else:
                    img[i][j] = -(img[i][j] + img_ap[i][j] - 255)

            # if ed < 0:
            #     img[i][j] = -img[i][j]
            img_bp[i][j] = img_ap[i][j] + img[i][j]

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


def Encode(img_path, predict_dir, arithmetic_dir):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)

    img_1 = b
    img_1 = predictive2(img_1)

    img_2 = g
    img_2 = predictive2(img_2)

    img_3 = r
    img_3 = predictive2(img_3)

    img = cv2.merge([img_1, img_2, img_3])
    f = os.path.split(img_path)[1]
    img_num = f[0:f.rfind('.png')]
    img_predict_path = os.path.join(predict_dir, img_num) + '.bmp'
    cv2.imwrite(img_predict_path, img)
    img_arithmetic_path = os.path.join(arithmetic_dir, img_num) + '.new'
    os.system('python adaptive-arithmetic-compress.py %s %s' % (img_predict_path, img_arithmetic_path))
    return img_arithmetic_path, img_num


def Decode(img_arithmetic_path, img_num, after_arithmetic_dir):
    img_after_arithmetic_path = os.path.join(after_arithmetic_dir, img_num) + '.bmp'
    os.system('python adaptive-arithmetic-decompress.py %s %s' % (img_arithmetic_path, img_after_arithmetic_path))
    img = cv2.imread(img_after_arithmetic_path)
    b, g, r = cv2.split(img)

    img_1 = b
    img_1 = anti_predicting(img_1)
    img_2 = g
    img_2 = anti_predicting(img_2)
    img_3 = r
    img_3 = anti_predicting(img_3)

    img = cv2.merge([img_1, img_2, img_3])
    return img


def main(img_path):
    predict_dir = 'predict'
    arithmetic_dir = 'arithmetic'
    after_arithmetic_dir = 'arithmetic_decode'
    img_arithmetic_path, img_num = Encode(img_path, predict_dir, arithmetic_dir)
    img_reconstruct = Decode(img_arithmetic_path, img_num, after_arithmetic_dir)

    img_original = cv2.imread(img_path)
    error_rate = fidelity(img_original, img_reconstruct)

    original_size = img_original.shape[0] * img_original.shape[1] * 3
    compress_size = os.path.getsize(img_arithmetic_path)
    return original_size / compress_size
