from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
import torch
from prefetch_generator import BackgroundGenerator
import h5py


class HDR_Dataset(Dataset):
    def __init__(self, datadir, crop_size=256):
        hf = h5py.File(datadir)
        self.inputs = hf.get('IN')
        self.labels = hf.get('GT')
        self.num = self.inputs.shape[0]
        self.h = self.inputs.shape[1]
        self.w = self.inputs.shape[2]
        self.crop_size = crop_size

    def __getitem__(self, index):
        in_imgs = self.inputs[index, :, :, :]
        ref_HDR = self.labels[index, :, :, :]
        # in_imgs, ref_HDR = self.random_crop(in_imgs, ref_HDR, crop_size=self.crop_size)
        in_imgs, ref_HDR = self.augment(in_imgs, ref_HDR, h=self.crop_size)
        return torch.from_numpy(in_imgs), torch.from_numpy(ref_HDR)

    def __len__(self):
        return self.num

    # def random_crop(self, in_imgs, ref_HDR, crop_size):
    #     crop_h_start = random.randint(0, self.h - crop_size - 1)
    #     crop_w_start = random.randint(0, self.w - crop_size - 1)
    #     in_imgs = in_imgs[:, crop_h_start: crop_h_start + crop_size, crop_w_start: crop_w_start + crop_size]
    #     ref_HDR = ref_HDR[:, crop_h_start: crop_h_start + crop_size, crop_w_start: crop_w_start + crop_size]
    #     return in_imgs, ref_HDR

    def augment(self, in_imgs, ref_HDR, h):
        in_imgs = np.transpose(in_imgs, axes=(1, 2, 0))
        ref_HDR = np.transpose(ref_HDR, axes=(1, 2, 0))
        flip = random.randint(0, 1)
        if flip == 1:
            in_imgs = cv2.flip(in_imgs, 1)
            ref_HDR = cv2.flip(ref_HDR, 1)

        rotate = random.randint(0, 3)
        mat = cv2.getRotationMatrix2D((h // 2, h // 2), 90*rotate, 1)
        in_imgs = cv2.warpAffine(in_imgs, mat, (h, h))
        ref_HDR = cv2.warpAffine(ref_HDR, mat, (h, h))

        in_imgs = np.transpose(in_imgs, axes=(2, 0, 1))
        ref_HDR = np.transpose(ref_HDR, axes=(2, 0, 1))
        return in_imgs, ref_HDR


class HDR_DataLoader_pre(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
