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
import torch.nn.functional as F
import torch
import torch.nn as nn
from unet3d import UNet3D
import time
import numpy as np
from segment_image import segment
from data_prep import load_nifty, save_nifty


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


if __name__ == '__main__':
    create_struct_seg_quarter_segmentations()

