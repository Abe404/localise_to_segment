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
from os.path import join
import torch.nn.functional as F
import torch
import torch.nn as nn
from unet3d import UNet3D
import time
import numpy as np
from skimage import measure
from segment_image import segment
from data_prep import load_nifty, save_nifty, load_im_and_heart
from datasets import create_train_val_test_split
from metrics import (get_metrics_from_arrays,
                     get_metrics_str, get_metrics,
                     get_metric_csv_row,
                     get_metric_header_str)


def create_struct_seg_quarter_segmentations():
    """
    Get segmentations using the localisation network for the best model.
    This will later be used for restricting to the single largest region.
    """
    data_dir = 'data/ThoracicOAR_quarter'
    all_fnames = os.listdir(data_dir)
    patch_shape = (48, 128, 128)

    exp_output_dir = 'train_output/struct_seg_heart_quarter_30/runs/0'
    model_path = os.path.join(exp_output_dir, 'models',
                              '23_epoch_109_dice_0.8937')
    
    seg_dir = os.path.join(exp_output_dir, 'seg')

    # create seg dir if it doesn't exist for this training run output.
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)

    # Model class must be defined somewhere
    cnn = torch.load(model_path).cuda()
    cnn = nn.DataParallel(cnn)
    cnn.eval()

    for dir_name in os.listdir(data_dir):
        im_path = os.path.join(data_dir, dir_name, 'data.nii.gz')
        im = load_nifty(im_path)
        # torch needs float32
        im = im.astype(np.float32)
        preds = segment(cnn, im, 2, patch_shape, patch_shape)
        # create folder with unique name for this image.
        seg_im_dir = os.path.join(seg_dir, dir_name)
        # then save the segmentation in this parent folder.
        # this is in order to keep the structure similar to struct seg so
        # other utilities can be used.
        if not os.path.isdir(seg_im_dir):
            os.makedirs(seg_im_dir)
        print('preds unique = ', np.unique(preds))
        out_path = os.path.join(seg_im_dir, 'label.nii.gz')
        print('saving segmentation to', out_path)
        save_nifty(preds, out_path)


def get_full_res_cropped_by_net_performance():
    """ take the images that were cropped by the network
        And segment them using the model that was trained with early stopping.
        This is a process we can run on the test set. """

    data_dir = 'data/ThoracicOAR'
    quarter_dir = 'data/ThoracicOAR_quarter'
    all_fnames = os.listdir(data_dir)
    model_path = 'train_output/struct_seg_heart_cropped/runs/1/models/17_epoch_47_dice_0.9366'
    fine_cnn = torch.load(model_path).cuda()
    fine_cnn = nn.DataParallel(fine_cnn)
    fine_cnn.eval()

    patch_shape = (64, 256, 256)
    tps = []
    tns = []
    fps = []
    fns = []
    quarter_seg_dir = 'train_output/struct_seg_heart_quarter_30/runs/0/seg'

    _, val, _ = create_train_val_test_split(data_dir)

    # go through each image and annotation in the full res validation dataset
    for im_name in val:
        
        full_res_im, full_res_annot = load_im_and_heart(join(data_dir, im_name))

        # get single largest region as predicted by course network,
        seg_path = os.path.join(quarter_seg_dir, im_name, 'label.nii.gz')
        seg = load_nifty(seg_path).astype(np.int16)
        labels = measure.label(seg == 1, background=0)
        largest_label = None

        # ignore background
        unique_labels = [l for l in np.unique(labels) if l > 0]
        label_sums = [np.sum(labels == l) for l in unique_labels]

        # restrict to biggesr region (the heart hopefully)
        label_sum, largest_fg_label = sorted(zip(label_sums,
                                                 unique_labels), reverse=True)[0]

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
     
        # then get the location of the heart in the full res 
        # version using the ratio for each axis.
        z_mid = z_mid * full_res_im.shape[0]
        y_mid = y_mid * full_res_im.shape[1]
        x_mid = x_mid * full_res_im.shape[2]

        z_mid = round(z_mid)
        y_mid = round(y_mid)
        x_mid = round(x_mid)
        # for crop
        z_min = z_mid - (patch_shape[0] // 2)
        y_min = y_mid - (patch_shape[1] // 2)
        x_min = x_mid - (patch_shape[2] // 2)

        # for the full res predictions, set everything outside
        # the predicted region to 0
        preds = np.zeros(full_res_im.shape)
        cropped_im = full_res_im[z_min:z_min+patch_shape[0],
                                 y_min:y_min+patch_shape[1],
                                 x_min:x_min+patch_shape[2]].astype(np.float32)

        # get the predicted region using the cropped 3D network.
        cropped_segmented = segment(fine_cnn, cropped_im, 2,
                                    patch_shape, patch_shape)

        preds[z_min:z_min+patch_shape[0],
              y_min:y_min+patch_shape[1],
              x_min:x_min+patch_shape[2]] = cropped_segmented

        foregrounds_int = full_res_annot.reshape(-1).astype(np.int)
        preds_int = preds.reshape(-1).astype(np.int)

        m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        im_tps = (np.sum((foregrounds_int == 1) * (preds_int == 1)))
        im_tns = (np.sum((foregrounds_int == 0) * (preds_int == 0)))
        im_fps = (np.sum((foregrounds_int == 0) * (preds_int == 1)))
        im_fns = (np.sum((foregrounds_int == 1) * (preds_int == 0)))

        image_metrics = get_metrics(im_tps, im_fps, im_tns, im_fns)
        print('metrics for', im_name, get_metrics_str(image_metrics))
        
        tps.append(im_tps)
        tns.append(im_tns)
        fps.append(im_fps)
        fns.append(im_fns)

    m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
    print('dataset metrics', get_metrics_str(m))
        
if __name__ == '__main__':
    get_full_res_cropped_by_net_performance()
