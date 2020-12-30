"""
Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from data_prep import load_im_and_heart
import random


def create_train_val_test_split(data_dir):
    all_fnames = os.listdir(data_dir)
    # 34 in the train set and 8 in the validation set and 8 in the test set.
    random.seed(42)
    random.shuffle(sorted(all_fnames))
    total = len(all_fnames)
    num_train = round(total * 0.6)
    num_val = round(total * 0.2)
    train_set = all_fnames[:num_train]
    val_set = all_fnames[num_train:num_train+num_val]
    test_set = all_fnames[num_train+num_val:]
    assert len(train_set) == num_train
    assert len(val_set) == num_val
    assert len(test_set) == total - (num_train + num_val)
    return train_set, val_set, test_set
    

class ImageDataset(Dataset):
    """ Dataset of full images """

    def __init__(self, im_names, parent_folder_path, patch_shape=None):
        """ if patch_size is specified then take a random
            patch with this size from the image and annotation """
            
        self.im_names = im_names
        self.parent_dir = parent_folder_path
        self.path_shape = patch_shape

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        im, heart_labels = load_im_and_heart(os.path.join(self.parent_dir, im_name))
        
        if self.patch_shape:
            # Limits for possible sampling locations from image
            # (based on size of image)
            depth_lim = im.shape[0] - patch_shape[0]
            bottom_lim = im.shape[1] - patch_shape[1]
            right_lim = im.shape[2] - patch_shape[2]

            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            z_in = math.floor(random.random() * depth_lim)

            im = image[z_in:z_in+patch_shape[0],
                       y_in:y_in+patch_shape[1],
                       x_in:x_in+patch_shape[2]]

        # add a channel dimension at the start, torch needs it
        im = np.expand_dims(im, axis=0).astype(np.float32)
        heart_labels = heart_labels.astype(np.int16)
        return im, heart_labels


def create_datasets(data_dir, patch_shape=None):
    """ patch size is used for train images
        to return a random patch from each image """
    train, val, test = create_train_val_test_split(data_dir)
    train_dataset = ImageDataset(train, data_dir, patch_shape)
    val_dataset = ImageDataset(val, data_dir)
    test_dataset = ImageDataset(test, data_dir)
    return train_dataset, val_dataset, test_dataset
