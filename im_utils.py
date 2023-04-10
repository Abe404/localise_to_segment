"""
Copyright (C) 2023 Abraham George Smith

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
from skimage.exposure import rescale_intensity
from skimage.color import gray2rgb

from skimage import measure
from file_utils import load_nifty


def contrast_enhance(im):
    p2, p98 = np.percentile(im, (2, 98))
    im = rescale_intensity(im, in_range=(p2, p98))
    return im

def load_annot(annot_path):
    annot = load_nifty(annot_path)
    # for pancreas we treat both tumor (index=2) and pancreas (index=1) 
    # as the pancreas. Thus we assign everying above 0 to 1 
    annot[annot > 0] = 1
    return annot.astype(np.uint8) 

def load_image(image_path):
    image = load_nifty(image_path)
    # if the first part of the image shape is 2 then assume it is the  prostae
    # dataset from the medical image decathlon. 
    # in which case we want the first channel.
    # also we want to move the last channel (depth) to be first
    # as the annotation have depth first and so do all other dataset images.
    if image.shape[0] == 2:
        image = image[0]
        image = np.moveaxis(image, -1, 0) # make consistent with annotation
    return image


def create_debug_image(im, annot):
    # Checking that pre-processing was ran correctly
    z, y, x = get_organ_centroid(annot)
    z = int(z)
    y = int(y)
    x = int(x)
    # scale image to 0-1 range
    central_x_slice_im = contrast_enhance(im[z, :, :])
    central_x_slice_labels = annot[z, :, :]
    #central_y_slice_im = contrast_enhance(im[:, y, :])
    #central_y_slice_labels = annot[:, y, :]
    x_slice_rgb = gray2rgb(central_x_slice_im)
    x_slice_outline = np.array(x_slice_rgb)
    x_slice_outline[:, :, 0][central_x_slice_labels > 0] = np.max(central_x_slice_im)
    x_slice_outline[:, :, 1][central_x_slice_labels > 0] *= 0.5
    x_slice_outline[:, :, 2][central_x_slice_labels > 0] *= 0.5
    combined_x = np.hstack((x_slice_rgb, x_slice_outline))
    return combined_x


def get_organ_shape(seg):
    labels = measure.label(seg == 1, background=0)
    # ignore background
    unique_labels = [l for l in np.unique(labels) if l > 0]
    assert len(unique_labels) > 0, unique_labels
    label_sums = [np.sum(labels == l) for l in unique_labels]
    # restrict to biggesr region (the organ hopefully)
    _label_sum, largest_fg_label = sorted(zip(label_sums, unique_labels), reverse=True)[0]
    # restrict to single largest region
    seg[labels != largest_fg_label] = 0
    label_coords = np.argwhere(seg == 1)

    zs = label_coords[:, 0]
    ys = label_coords[:, 1]
    xs = label_coords[:, 2]

    depth = np.max(zs) - np.min(zs)
    height = np.max(ys) - np.min(ys)
    width = np.max(xs) - np.min(xs)
    return (depth, height, width)


def get_mean_shape(input_dir):
    """ Go through a folder of images.
        and see what.
    """
    im_dir_names = [f for f in os.listdir(input_dir) if f[0] != '.']
    depths = []
    heights = []
    widths = []
    for fname in im_dir_names:
        im_dir = os.path.join(input_dir, fname)
        annot = load_nifty(os.path.join(im_dir, 'label.nii.gz'))
        print('Annot shape', annot.shape)
        print('Annot sum', np.sum(annot))
        depths.append(annot.shape[0]) 
        heights.append(annot.shape[1]) 
        widths.append(annot.shape[2]) 
    return (np.mean(depths), np.mean(heights), np.mean(widths))


def get_organ_centroid(seg):
    labels = measure.label(seg == 1, background=0)
    # ignore background
    unique_labels = [l for l in np.unique(labels) if l > 0]
    assert len(unique_labels) > 0, unique_labels
    label_sums = [np.sum(labels == l) for l in unique_labels]
    # restrict to biggesr region (the organ hopefully)
    _label_sum, largest_fg_label = sorted(zip(label_sums, unique_labels), reverse=True)[0]
    # restrict to single largest region
    seg[labels != largest_fg_label] = 0
    label_coords = np.argwhere(seg == 1)

    zs = label_coords[:, 0]
    ys = label_coords[:, 1]
    xs = label_coords[:, 2]

    depth = np.max(zs) - np.min(zs)
    height = np.max(ys) - np.min(ys)
    width = np.max(xs) - np.min(xs)
    z_mid = np.min(zs) + (depth / 2)
    y_mid = np.min(ys) + (height / 2)
    x_mid = np.min(xs) + (width / 2)
    return z_mid, y_mid, x_mid    



def get_crop_coords_to_organ(seg, padding=0):
    labels = measure.label(seg == 1, background=0)
    # ignore background
    unique_labels = [l for l in np.unique(labels) if l > 0]
    label_sums = [np.sum(labels == l) for l in unique_labels]
    # restrict to biggesr region (the organ hopefully)
    _label_sum, largest_fg_label = sorted(zip(label_sums, unique_labels),
                                         reverse=True)[0]
    # restrict to single largest region
    seg[labels != largest_fg_label] = 0
    label_coords = np.argwhere(seg == 1)

    zs = label_coords[:, 0]
    ys = label_coords[:, 1]
    xs = label_coords[:, 2]

    x_min = np.min(xs) - padding
    y_min = np.min(ys) - padding
    z_min = np.min(zs) - padding

    x_max = np.max(xs) + padding
    y_max = np.max(ys) + padding
    z_max = np.max(zs) + padding

    # round to integers as will be used for cropping voxels
    return (round(z_min), round(y_min), round(x_min),
            round(z_max), round(y_max), round(x_max))
