# https://www.jianshu.com/p/43d548ad6b5d

import argparse
import glob
import os
import re

import numpy as np
import tensorflow as tf

import cv2
from skimage.measure import compare_ssim as ssim

# 均方误差MSE
def mse(image_original, image_created):
    error = np.sum((image_original.astype("float") - image_created.astype("float")) ** 2)    
    error /= float(image_original.shape[0] * image_created.shape[1])
    return error

def psnr_block(image_original_temp, image_created_temp):
    def read_img(path):
        return tf.image.decode_image(tf.read_file(path))
    
    def psnr(tf_img1, tf_img2):
        return tf.image.psnr(tf_img1, tf_img2, max_val=255)
    
    def _main():
        t1 = read_img(image_original_temp)
        t2 = read_img(image_created_temp)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(psnr(t1, t2))
            print(y)

# 即m×n单色图像 I 和 K（原图像与处理图像）之间均方误差
# def MSE(I, K):
#     x, y = tf.cast(I, tf.float32), tf.cast(K, tf.float32)
#     mse = tf.losses.mean_squared_error(labels=y, predictions=x)
#     return mse

# 结构相似性SSIM

# def _tf_fspecial_gauss(size, sigma):
#     """Function to mimic the 'fspecial' gaussian MATLAB function"""
#     x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

#     x_data = np.expand_dims(x_data, axis=-1)
#     x_data = np.expand_dims(x_data, axis=-1)

#     y_data = np.expand_dims(y_data, axis=-1)
#     y_data = np.expand_dims(y_data, axis=-1)

#     x = tf.constant(x_data, dtype=tf.float32)
#     y = tf.constant(y_data, dtype=tf.float32)

#     g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#     return g / tf.reduce_sum(g)


# def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma)    # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     L = 1  # depth of image (255 in case the image has a different scale)
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
#     if cs_map:
#         value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
#                                                     (sigma1_sq + sigma2_sq + C2)),
#                 (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
#     else:
#         value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
#                                                     (sigma1_sq + sigma2_sq + C2))

#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value


# def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
#     weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
#     mssim = []
#     mcs = []
#     for l in range(level):
#         ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
#         mssim.append(tf.reduce_mean(ssim_map))
#         mcs.append(tf.reduce_mean(cs_map))
#         filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#         filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#         img1 = filtered_im1
#         img2 = filtered_im2

#     list to tensor of dim D+1
#     mssim = tf.stack(mssim, axis=0)
#     mcs = tf.stack(mcs, axis=0)

#     value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*(mssim[level-1]**weight[level-1]))

#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value

# 峰值信噪比PSNR
def PSNR(I, K):
    x, y = tf.cast(I, tf.float32), tf.cast(K, tf.float32)
    mse = tf.losses.mean_squared_error(labels=y, predictions=x)
    psnr = 10*tf.log(255**2/mse)/tf.log(10)
    return psnr

def argument():
    parser = argparse.ArgumentParser(
        usage = "To evaluate the similarity of images which are created and original."
        )
    parser.add_argument("--dir-image-created", type=str, help="dir of images created.")
    parser.add_argument("--dir-image-original", type=str, help="dir of images original.")
    parser.add_argument("--step", type=int, help="step between the two images which are created and original.")
    args = parser.parse_args()

    return args

def main(args):

    path_image_created = os.path.join(args.dir_image_created, "*")
    list_image_created = glob.glob(path_image_created)
    list_image_created_sorted = sort_humanly(list_image_created)

    path_image_original = os.path.join(args.dir_image_original, "*")
    list_image_original = glob.glob(path_image_original)
    list_image_original_sorted = sort_humanly(list_image_original)

    for index in range(0, len(list_image_created_sorted), 1):
        image_created_temp = cv2.imread(list_image_created_sorted[index], flags=2)
        image_original_temp = cv2.imread(list_image_original_sorted[index + args.step], flags=2)

        # 均方误差MSE
        result_mse = mse(image_original_temp, image_created_temp)
        result_mse = mse(image_original_temp, image_original_temp)
        result_mse = mse(image_created_temp, image_created_temp)
        
        # 结构相似性SSIM(相似度指数)
        result_ssim = ssim(image_original_temp, image_created_temp)
        result_ssim = ssim(image_original_temp, image_original_temp)
        result_ssim = ssim(image_created_temp, image_created_temp)
        # result_ssim = tf_ms_ssim(image_original_temp, image_created_temp)
        # print(result_ssim)

        # 峰值信噪比PSNR
        result_psnr = psnr_block(image_original_temp, image_created_temp)
        print(result_psnr)

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):
    return sorted(v_list, key=str2int)

if __name__ == '__main__':
    args = argument()
    main(args)

    print("evaluation")
