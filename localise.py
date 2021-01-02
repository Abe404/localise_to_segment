"""
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
import os
import numpy as np
import nibabel as nib
from skimage.io import imread
from skimage import measure
from skimage import filters
from data_prep import load_nifty
from data_prep import save_nifty

# cycle through all the images
input_dir = os.path.join('data', 'ThoracicOAR')
input_seg_dir='train_output/struct_seg_heart_quarter_30/runs/0/seg'
im_dir_names = os.listdir(input_dir)
output_shape = (64,256,256)

# cropped dataset using nework.
output_dir = os.path.join('data', 'ThoracicOAR_cropped_using_network')

for im_name in im_dir_names:
    # get the location of the heart in the quarter res version
    seg_path = os.path.join(input_seg_dir, im_name, 'label.nii.gz')
    seg = load_nifty(seg_path).astype(np.int16)
    labels = measure.label(seg == 1, background=0)
    largest_label = None
    # ignore background
    unique_labels = [l for l in np.unique(labels) if l > 0]
    label_sums = [np.sum(labels == l) for l in unique_labels]
    # restrict to biggesr region (the heart hopefully)
    label_sum, largest_fg_label = sorted(zip(label_sums, unique_labels), reverse=True)[0]
    # restrict to single largest region
    seg[labels != largest_fg_label] = 0
    label_coords = np.argwhere(seg == 1)

    zs = label_coords[:, 0]
    ys = label_coords[:, 1]
    xs = label_coords[:, 2]

    z_mid = np.min(zs) + (np.max(zs) - np.min(zs))
    y_mid = np.min(ys) + (np.max(ys) - np.min(ys))
    x_mid = np.min(xs) + (np.max(xs) - np.min(xs))

    # get this as a ratio i.e 0-1
    z_mid = z_mid / seg.shape[0]
    y_mid = y_mid / seg.shape[1]
    x_mid = x_mid / seg.shape[2]
   
    # now load the full res image.
    full_res_path = os.path.join(input_dir, im_name, 'data.nii.gz')
    full_res_im = load_nifty(full_res_path)

    # then get the location of the heart in the full res 
    # version using the ratio for each axis.
    z_mid = z_mid * full_res_im.shape[0]
    y_mid = y_mid * full_res_im.shape[1]
    x_mid = x_mid * full_res_im.shape[2]

    z_mid = round(z_mid)
    y_mid = round(y_mid)
    x_mid = round(x_mid)

    # Then get the cropped region from the full res image
    # save in the output folder.

    # if the folder doesn't exist then create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cropped_im_data_dir = os.path.join(output_dir, im_name)
    if not os.path.isdir(cropped_im_data_dir):
        os.makedirs(cropped_im_data_dir)

    z_min = z_mid - (output_shape[0] // 2)
    y_min = y_mid - (output_shape[1] // 2)
    x_min = x_mid - (output_shape[2] // 2)

    cropped_im = full_res_im[z_min:z_min+output_shape[0],
                             y_min:y_min+output_shape[1],
                             x_min:x_min+output_shape[2]]

    assert cropped_im.shape == output_shape

    # cropped dataset using nework.
    output_im_dir = os.path.join(output_dir, im_name)
    if not os.path.isdir(output_im_dir):
        os.makedirs(output_im_dir)

    output_path = os.path.join(output_im_dir, 'data.nii.gz')
    print('saving cropped image to ', output_path) 
    save_nifty(cropped_im, output_path)
