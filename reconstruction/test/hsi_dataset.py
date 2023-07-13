from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import hdf5storage
from imageio import imread

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 512,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('.mat','_RGB_D_gc.png') for line in hyper_list]
            nir_list = [line.replace('.mat','_NIR.jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(bgr) of MobiSpectral dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            hyper = cube['rad'][:,:,1:204:3]
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            nir_path = bgr_data_path + nir_list[i]
            bgr = imread(bgr_path)
            nir = imread(nir_path)
            bgr = np.float32(bgr)
            nir = np.float32(nir)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            nir = (nir-nir.min())/(nir.max()-nir.min())
            bgr = np.dstack((bgr,nir))
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'MobiSpectral scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/test_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('.mat','_RGB_D_gc.png') for line in hyper_list]
            nir_list = [line.replace('.mat','_NIR.jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        nir_list.sort()
        print(f'len(hyper_valid) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of MobiSpectral dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            hyper = cube['rad'][:,:,1:204:3]
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            nir_path = bgr_data_path + nir_list[i]
            bgr = imread(bgr_path)
            nir = imread(nir_path)
            bgr = np.float32(bgr)
            nir = np.float32(nir)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            nir = (nir - nir.min()) / (nir.max() - nir.min())

            bgr = np.dstack((bgr,nir))
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'MobiSpectral scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
