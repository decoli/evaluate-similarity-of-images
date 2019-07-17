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

def psnr_block(image_original_temp, image_created_temp, index, number_image):
    def read_img(path):
        return tf.image.decode_image(tf.read_file(path))
    
    def psnr(tf_img1, tf_img2):
        return tf.image.psnr(tf_img1, tf_img2, max_val=255)
    
    def _main(image_original_temp, image_created_temp):
        t1 = read_img(image_original_temp)
        t2 = read_img(image_created_temp)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(psnr(t1, t2))
            print("{}--[{}]--{:.2%}--{}".format("psnr", index, float((index + 1) / number_image) , y))

    _main(image_original_temp, image_created_temp)

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
        if index + args.step > len(list_image_original_sorted) - 1:
            break

        image_created_temp = cv2.imread(list_image_created_sorted[index], flags=2)
        image_original_temp = cv2.imread(list_image_original_sorted[index + args.step], flags=2)

        # 均方误差MSE
        result_mse = mse(image_original_temp, image_created_temp)
        # result_mse = mse(image_original_temp, image_original_temp)
        # result_mse = mse(image_created_temp, image_created_temp)
        
        # 结构相似性SSIM(相似度指数)
        result_ssim = ssim(image_original_temp, image_created_temp)
        # result_ssim = ssim(image_original_temp, image_original_temp)
        # result_ssim = ssim(image_created_temp, image_created_temp)

        # 峰值信噪比PSNR
        result_psnr = psnr_block(list_image_original_sorted[index + args.step], list_image_created_sorted[index], index=index, number_image=len(list_image_created_sorted))

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
