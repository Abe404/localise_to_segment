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
import random
import numpy as np
import nibabel as nib

from skimage.io import imread
from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize
from skimage import img_as_float
import shutil

random.seed(42)

def save_nifty(image, image_path):
    """ save compressed nifty file to disk and
        switch first and last channel
    """
    image = np.moveaxis(image, 0, -1) # depth moved to end
    img = nib.Nifti1Image(image, np.eye(4))
    img.to_filename(image_path)


def load_nifty(image_path):
    """ load compressed nifty file from disk and
        switch first and last channel
    """
    image = nib.load(image_path)
    image = np.array(image.dataobj)
    image = np.moveaxis(image, -1, 0) # depth moved to beginning
    return image


def save_im_and_heart(data_dir, im, labels):
    """
    Used for saving the half resolution image
    """
    save_nifty(im, os.path.join(data_dir, 'data.nii.gz'))
    save_nifty(labels, os.path.join(data_dir, 'label.nii.gz'))


def load_im_and_heart(data_dir):
    """ Load a heart from the struct seg data.
        Include binary (0,1) labels for the voxels which contain the heart.
    """
    image = load_nifty(os.path.join(data_dir, 'data.nii.gz'))
    annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))
    filter_labels = len(np.unique(annot)) > 2

    if filter_labels:
        heart_labels = (annot == 3).astype(np.int16) # only heart
        return image, heart_labels

    return image, annot.astype(np.int16)


def get_mean_shape(input_dir):
    """ Go through a folder of images.
        and see what.
    """
    im_dir_names = os.listdir(input_dir)
    depths = []
    heights = []
    widths = []
    for im_name in im_dir_names:
        im_dir = os.path.join(input_dir, im_name)
        annot = load_nifty(os.path.join(im_dir, 'label.nii.gz'))
        print('Annot shape', annot.shape)
        print('Annot sum', np.sum(annot))
        depths.append(annot.shape[0]) 
        heights.append(annot.shape[1]) 
        widths.append(annot.shape[2]) 
    return (np.mean(depths), np.mean(heights), np.mean(widths))
        

def create_smaller_size_dataset(input_dir, output_dir, scale):
    im_dir_names = os.listdir(input_dir)
    # if the folder doesn't exist then create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # first get the mean shape of the input images.
    mean_input_shape = get_mean_shape(input_dir)
    # depth is scaled twice. It is already undersampled.
    scaled_shape = (2 * round(mean_input_shape[0] * scale),
                    round(mean_input_shape[1] * scale),
                    round(mean_input_shape[2] * scale))
    print('mean input shape', mean_input_shape)
    print('scaled shape', scaled_shape)
    for im_name in im_dir_names:
        full_im_data_dir = os.path.join(input_dir, im_name)
        scaled_im_data_dir = os.path.join(output_dir, im_name)
        print('checking', im_name, end=',')
        if not os.path.isdir(scaled_im_data_dir):
            os.makedirs(scaled_im_data_dir)
            # get data and heart labels for a patch with the heart in it
            image_data, heart_labels = load_im_and_heart(full_im_data_dir)
            total_heart = np.sum(heart_labels)
            total_scan = len(heart_labels.reshape(-1))
            percent_heart = (total_heart / total_scan)
            print('Loading heart image,',
                  'actual', total_heart,
                  'out of total', total_scan,
                   'giving percent', 100 * percent_heart)
            scaled_im_data = resize(image_data, scaled_shape)
            scaled_im_labels = resize(heart_labels.astype(np.float64),
                                      scaled_shape)
            scaled_im_labels = np.round(scaled_im_labels).astype(np.int16)
            print('heart labels to save sum = ', np.sum(scaled_im_labels))
            print('heart labels to save unique = ', np.unique(scaled_im_labels))
            save_im_and_heart(scaled_im_data_dir,
                              scaled_im_data,
                              scaled_im_labels)

if __name__ == '__main__':
    input_dir = os.path.join('data', 'ThoracicOAR')
    # output_dir = os.path.join('data', 'ThoracicOAR_half')
    # output_dir = os.path.join('data', 'ThoracicOAR_eighth')
    output_dir = os.path.join('data', 'ThoracicOAR_quarter')
    #create_smaller_size_dataset(input_dir, output_dir, 1/8)
    create_smaller_size_dataset(input_dir, output_dir, 1/4)
    # mean_input_shape = get_mean_shape(output_dir)
