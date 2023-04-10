"""
Copyright (C) 2020 Abraham George Smith
Copyright (C) 2021 Abraham George Smith

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

import random
import math
import numpy as np
import im_utils
from torch.utils.data import Dataset

def create_train_val_test_split(image_fpaths, annot_fpaths):
    # 60% in train. 20% in val and 20% in test
    random.seed(42)
    assert len(image_fpaths) == len(annot_fpaths)
    zipped = list(zip(image_fpaths, annot_fpaths))
    random.shuffle(zipped)
    image_fpaths, annot_fpaths = zip(*zipped)
    # list come out as tuples, and so must be converted to lists.
    image_fpaths, annot_fpaths = list(image_fpaths), list(annot_fpaths)

    total = len(image_fpaths)
    num_train = round(total * 0.6)
    num_val = round(total * 0.2)


    image_fpaths_train = image_fpaths[:num_train]
    image_fpaths_val = image_fpaths[num_train:num_train+num_val]
    image_fpaths_test = image_fpaths[num_train+num_val:]


    annot_fpaths_train = annot_fpaths[:num_train]
    annot_fpaths_val = annot_fpaths[num_train:num_train+num_val]
    annot_fpaths_test = annot_fpaths[num_train+num_val:]

    assert len(image_fpaths_train) == num_train
    assert len(image_fpaths_val) == num_val
    assert len(image_fpaths_test) == total - (num_train + num_val)

    return (image_fpaths_train, image_fpaths_val, image_fpaths_test,
            annot_fpaths_train, annot_fpaths_val, annot_fpaths_test)
    

class ImageDataset(Dataset):
    """ Dataset of full images """

    def __init__(self, image_fpaths, annot_fpaths, raw_ds,
                 patch_shape=None, organ=None, force_fg_percent=None):
        """ if patch_size is specified then take a random
            patch with this size from the image and annotation """
        self.image_fpaths = image_fpaths
        self.annot_fpaths = annot_fpaths
        self.patch_shape = patch_shape
        self.force_fg_percent = force_fg_percent
        self.organ = organ
        self.raw_ds = raw_ds

    def __len__(self):
        return len(self.image_fpaths)

    def __getitem__(self, idx):

        image_fpath = self.image_fpaths[idx]
        annot_fpath = self.annot_fpaths[idx]

        im = im_utils.load_image(image_fpath)
        annot = im_utils.load_annot(annot_fpath)

        # don't need to sample patch if patch shape not supplied.
        # This is used for validation and segmenation of full images.
        if not self.patch_shape:
            # add a channel dimension at the start, torch needs it
            im = np.expand_dims(im, axis=0).astype(np.float32)
            annot = annot.astype(np.int16)
            return im, annot

        found_patch = False
        fg_required = False
        # if we should force some patches to contain foreground.
        # then, for this percent of the time, ensure that the annotation
        # contains some fg
        tries = 0
        if self.force_fg_percent is not None:
            random_percent = random.random() * 100
            fg_required = random_percent < self.force_fg_percent
        while not found_patch:
            tries += 1
            # Limits for possible sampling locations from image
            # (based on size of image)
             
            depth_lim = im.shape[0] - min(im.shape[0], self.patch_shape[0])
            bottom_lim = im.shape[1] - min(im.shape[1], self.patch_shape[1])
            right_lim = im.shape[2] - min(im.shape[2], self.patch_shape[2])
            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            z_in = math.floor(random.random() * depth_lim)

            annot_patch = annot[z_in:z_in+self.patch_shape[0],
                                y_in:y_in+self.patch_shape[1],
                                x_in:x_in+self.patch_shape[2]]

            if not fg_required or np.any(annot_patch):
                found_patch = True

            im_patch = im[z_in:z_in+self.patch_shape[0],
                          y_in:y_in+self.patch_shape[1],
                          x_in:x_in+self.patch_shape[2]]
        # add a channel dimension at the start, torch needs it
        im_patch = np.expand_dims(im_patch, axis=0).astype(np.float32)
        annot_patch = annot_patch.astype(np.int16)
        return im_patch, annot_patch


def create_datasets(image_fpaths, annot_fpaths, raw_ds,
                    patch_shape=None, organ=None, force_fg_percent=None):
    """ patch size is used for train images
        to return a random patch from each image """
    (image_fpaths_train, image_fpaths_val,
     image_fpaths_test, annot_fpaths_train,
     annot_fpaths_val, annot_fpaths_test) = create_train_val_test_split(image_fpaths,
                                                                         annot_fpaths)

    train_dataset = ImageDataset(image_fpaths_train, annot_fpaths_train, raw_ds,
                                 patch_shape, organ, force_fg_percent)
    val_dataset = ImageDataset(image_fpaths_val, annot_fpaths_val, raw_ds, organ=organ)
    test_dataset = ImageDataset(image_fpaths_test, annot_fpaths_test, raw_ds, organ=organ)

    return train_dataset, val_dataset, test_dataset
