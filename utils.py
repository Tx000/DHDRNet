# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
# [2017-07] Modifications for sText2Image: Shangzhe Wu
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import cv2
import numpy as np


def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])
GAMMA = 2.2


def LDR2HDR(img, expo): # input/output 0~1
    return (img**GAMMA / expo)


def get_image2HDR(image_path, exp, image_size=None, is_crop=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(LDR2HDR(imread(image_path), exp), image_size, is_crop)


# always return RGB, float32, range -1~1
def get_image(image_path, image_size=None, is_crop=False, is_SR=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(imread(image_path), image_size, is_crop, is_SR)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path)/255.
    return img.astype(np.float32)[...,::-1]


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, path):
    if (path[-4:] == '.hdr'):
        #pdb.set_trace()
        return radiance_writer(path, images)
    else:
        return cv2.imwrite(path, images[...,::-1]*255.)


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)], (crop_w, crop_h))


def transform(image, image_size, is_crop, is_SR=False):
    # npx : # of pixels width/height of image
    if is_crop:
        out = center_crop(image, image_size)
    elif (image_size is not None):
        out = cv2.resize(image, image_size)
    else:
        out = image
    if is_SR:
        out = cv2.resize(out, (image_size[1] // 2, image_size[0] // 2))
    return out.astype(np.float32)


def inverse_transform(images):
    return (images+1.)/2.


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])