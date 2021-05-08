import os
import numpy as np
from PreProcess_m import dir_check
import cv2
import predict_arithmetic

os.system('python PreProcess_m.py')
input_dir = 'test'
before_ppm_dir = 'before_ppm'
jpg_dir = 'jpg'
after_ppm_dir = 'after_ppm'
jpls_decode_dir = 'decode'
jpeg2000_dir = 'jepg2000'
after_jpeg2000_dir = 'after_jpeg2000'
predict_dir = 'predict'
arithmetic_dir = 'arithmetic'
after_arithmetic_dir = 'arithmetic_decode'

jpeg_ls_ratio = []
png_ratio = []
jpeg2000_ratio = []
pa_total_ratio = []
num = 1

dir_check(before_ppm_dir, empty_flag=True)
dir_check(jpg_dir, empty_flag=True)
dir_check(after_ppm_dir, empty_flag=True)
dir_check(jpls_decode_dir, empty_flag=True)
dir_check(jpeg2000_dir, empty_flag=True)
dir_check(after_jpeg2000_dir, empty_flag=True)
dir_check(predict_dir, empty_flag=True)
dir_check(arithmetic_dir, empty_flag=True)
dir_check(after_arithmetic_dir, empty_flag=True)


for f in os.listdir(input_dir):
    img_name = os.path.join(input_dir, f)
    img = cv2.imread(img_name)

    # JPEG-LS
    before_ppm_name = os.path.join(before_ppm_dir, f[0:f.rfind('.png')]) + '.ppm'
    after_ppm_name = os.path.join(after_ppm_dir, f[0:f.rfind('.png')]) + '.ppm'
    cv2.imwrite(before_ppm_name, img)
    jpg_name = os.path.join(jpg_dir, f[0:f.rfind('.png')]) + '.jpg'
    os.system("jpeg -ls 2 -cls %s %s" % (before_ppm_name, jpg_name))
    os.system("jpeg -cls %s %s" % (jpg_name, after_ppm_name))
    img_decode = cv2.imread(after_ppm_name)
    img_decode_name = os.path.join(jpls_decode_dir, f)
    cv2.imwrite(img_decode_name, img_decode)
    # print((img == img_decode).all())
    original_size = img.shape[0] * img.shape[1] * 3
    size_jpg = os.path.getsize(jpg_name)
    j_ratio = original_size/size_jpg
    jpeg_ls_ratio.append(j_ratio)

    # PNG
    size_png = os.path.getsize(img_name)
    p_ratio = original_size / size_png
    png_ratio.append(p_ratio)

    # JPEG-2000
    img_name_header, NoUse = os.path.splitext(f)
    jp2_name = os.path.join(jpeg2000_dir, '%s.jp2' % img_name_header)
    # os.system("opj_compress -n 1 -i %s -o %s" % (img_name, jp2_name))
    os.system("opj_compress -i %s -o %s" % (img_name, jp2_name))
    size_jp2 = os.path.getsize(jp2_name)
    jp2_ratio = original_size / size_jp2
    jpeg2000_ratio.append(jp2_ratio)
    after_jpeg2000_name = os.path.join(after_jpeg2000_dir, '%s.png' % img_name_header)
    os.system("opj_decompress -i %s -o %s" % (jp2_name, after_jpeg2000_name))
    after_jpeg2000_img = cv2.imread(after_jpeg2000_name)
    is_same_jpeg2000 = (img == after_jpeg2000_img).all()

    # Predictive_Arithmetic
    pa_ratio = predict_arithmetic.main(img_name)
    pa_total_ratio.append(pa_ratio)

    print('Number:%d, the average compression ratio of JPEG-LS is %0.3f, minimum is %0.3f,'
          ' maximum is %0.3f,'
          ' variance is %0.4f; PNG: mean is %0.3f, minimum is %0.3f, maximum is %0.3f, variance is %0.4f; JPEG2000: '
          'mean is %0.3f, minimum is %0.3f, maximum is %0.3f, variance is %0.4f; Predict_Arithmetic: mean is %0.3f, '
          'minimum is %0.3f, maximum is %0.3f,variance is %0.4f '
          % (num, np.mean(jpeg_ls_ratio), min(jpeg_ls_ratio), max(jpeg_ls_ratio),
             np.var(jpeg_ls_ratio), np.mean(png_ratio), min(png_ratio), max(png_ratio), np.var(png_ratio),
             np.mean(jpeg2000_ratio), min(jpeg2000_ratio), max(jpeg2000_ratio), np.var(jpeg2000_ratio),
              np.mean(pa_total_ratio), min(pa_total_ratio), max(pa_total_ratio), np.var(pa_total_ratio)))

    num = num + 1
