import os
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from imageio import imread
import hdf5storage

def load_mat(mat_name, var_name):
        """ Helper function to load mat files (used in making h5 dataset) """
        data = hdf5storage.loadmat(mat_name, variable_names=[var_name])
        return data

class DatasetFromDirectory():
        IMAGE_SIZE = 8
        BLOCK_SIZE = 1
        row_itr, col_itr = 0, 0
        var_name = "cube"
        labels = []
        mats = []
        labelslist = ["organic","nonorganic"]
        loaded_mat = np.array((IMAGE_SIZE, IMAGE_SIZE, 51))

        def __init__(self, root, dataset_dir, fruit):
                self.root = root
                if fruit == 'kiwi' or fruit == 'apple':
                    self.IMAGE_SIZE = 8
                elif fruit == 'tomato':
                    self.IMAGE_SIZE = 32
                elif fruit == 'strawberries' or fruit == 'blueberries':
                    self.IMAGE_SIZE = 16
                self.fruit = fruit

                for directory in glob(os.path.join(root, dataset_dir, "*")):
                        for filename in glob(os.path.join(directory, "*.mat")):
                                label = filename.split("/")[-2].split("_")[1]
                                mat_file_name = filename.split("/")[-1].split("_")[0]
                                mat_file_name = mat_file_name  + '_RGB_D.mat'
                                for _ in range(self.IMAGE_SIZE//self.BLOCK_SIZE):
                                        for _ in range(self.IMAGE_SIZE//self.BLOCK_SIZE):
                                                self.labels.append(label)
                                                self.mats.append(os.path.join(directory, mat_file_name))

        def get_signature(self, mat):
            return mat[self.row_itr, self.col_itr,:]

        def get_mat(self, index, var_name, fruit):
                mat = load_mat(index, var_name)
                if fruit == 'kiwi' or fruit == 'apple':
                    # kiwi, apple
                    mat = mat[var_name][208:272:8,288:352:8,:]
                elif fruit == 'tomato':
                    # tomato
                    mat = mat[var_name][176:304:4,256:384:4,:]
                elif fruit == 'strawberries':
                    #strawberries
                    mat = mat[var_name][208:272:4,288:352:4,:]
                elif fruit == 'blueberries':
                    # blueberries
                    mat = mat[var_name][232:248,312:328,:]
                return mat

        def divide_mat(self, mat):
            return mat[self.row_itr:(self.row_itr+self.BLOCK_SIZE), self.col_itr:(self.col_itr+self.BLOCK_SIZE),:]

        def __len__(self):
                return len(self.labels)

        def __getitem__(self, index):
                if self.row_itr == 0 and self.col_itr == 0:
                        self.loaded_mat = self.get_mat(self.mats[index], self.var_name, self.fruit)
                signature = self.divide_mat(self.loaded_mat)
                self.row_itr += self.BLOCK_SIZE
                if self.row_itr == self.IMAGE_SIZE:
                       self.row_itr = 0
                       self.col_itr += self.BLOCK_SIZE
                       if self.col_itr == self.IMAGE_SIZE:
                               self.col_itr = 0
                label = self.labels[index]
                label = self.labelslist.index(label)
                return signature, label
