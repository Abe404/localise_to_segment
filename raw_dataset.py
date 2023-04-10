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
from file_utils import save_nifty
import im_utils
from im_utils import load_image, load_annot
from skimage.io import imsave
from skimage.transform import resize



# the class is used to represent all datasets consistently  in the project
class RawDataset:
    def __init__(self, im_dir, annot_dir, patch_size, organ_padding, batch_size=None, name=None):
        self.im_dir = im_dir
        self.annot_dir = annot_dir
        # low_res images are downsized and used for initial localisation network.
        self.low_res_image_dir = self.im_dir + '_low_res' 
        self.low_res_annot_dir = self.annot_dir + '_low_res' 

        self.debug_im_dir = os.path.join(os.path.dirname(self.im_dir), 'debug_images')
        self.crop_im_dir = self.im_dir + '_cropped'
        self.crop_annot_dir = self.annot_dir + '_cropped'
        self.patch_size = patch_size
        self.organ_padding = organ_padding # voxels to pad cropped organ
        self.batch_size = batch_size # batch size to use for training
        self.name = name # used for exp output dir

    def get_full_image_path(self, fname):
        return os.path.join(self.im_dir, fname)

    def get_full_annot_path(self, fname):
        return os.path.join(self.annot_dir, fname)

    def get_low_res_image_path(self, fname):
        return os.path.join(self.low_res_image_dir, fname)

    def get_low_res_annot_path(self, fname):
        return os.path.join(self.low_res_annot_dir, fname)

    def get_cropped_image_path(self, fname):
        return os.path.join(self.crop_im_dir, fname)

    def get_cropped_annot_path(self, fname):
        return os.path.join(self.crop_annot_dir, fname)

    def get_all_fnames(self):
        # get names of all files in this dataset as a list.
        return [f for f in sorted(os.listdir(self.im_dir)) if f[0] != '.']


    def print_largest_shapes(self):
        """ implemented to check that all organs fit within the 
            GPU patch size """
        
        depths = []
        widths = []
        heights = []
        fnames = sorted(self.get_all_fnames())
        for f in fnames:
            full_annot_fpath = self.get_cropped_annot_path(f)
            full_annot = load_annot(full_annot_fpath)
            depth, height, width = im_utils.get_organ_shape(full_annot)
            print('checking', f, depth, height, width)
            depths.append(depth)
            heights.append(height)
            widths.append(width)
        print('max depth', max(depths), fnames[np.argmax(depths)])
        print('max height', max(heights), fnames[np.argmax(heights)])
        print('max width', max(widths), fnames[np.argmax(widths)])


    def save_debug_image(self, fname, orig_im, annot, label_indices=[1]):
        if not os.path.isdir(self.debug_im_dir):
            os.makedirs(self.debug_im_dir)
        
        for organ_idx in label_indices:
            if organ_idx > 0:
                organ_annot = (annot == organ_idx)
                debug_im = im_utils.create_debug_image(orig_im, organ_annot)
                imsave(os.path.join(self.debug_im_dir, f'{fname}_{organ_idx}_x.png'), debug_im)

    def create_low_res_dataset(self):
        fnames = sorted(self.get_all_fnames())
        # if the folders dont exist then create them
        if not os.path.isdir(self.low_res_image_dir):
            os.makedirs(self.low_res_image_dir)
        if not os.path.isdir(self.low_res_annot_dir):
            os.makedirs(self.low_res_annot_dir)

        for fname in fnames:
            print('checking', fname, end=',')
            if not os.path.isdir(self.low_res_image_dir):
                os.makedirs(self.low_res_image_dir)
            # get data and organ labels for a patch with the organ in it
            image_fpath = self.get_full_image_path(fname)
            annot_fpath = self.get_full_annot_path(fname)
            image_data = load_image(image_fpath)
            
            scaled_shape = (image_data.shape[0] // 2,
                            image_data.shape[1] // 2,
                            image_data.shape[2] // 3)

            organ_labels = load_annot(annot_fpath)

            total_organ = np.sum(organ_labels)
            total_scan = len(organ_labels.reshape(-1))
            percent_organ = (total_organ / total_scan)

            print('Loading image,',
                  'actual', total_organ,
                  'out of total', total_scan,
                   'giving percent', 100 * percent_organ)

            low_res_image = resize(image_data, scaled_shape)
            low_res_annot = resize(organ_labels.astype(np.float64), scaled_shape)
            low_res_annot = np.round(low_res_annot).astype(np.int16)

            print('organ labels to save sum = ', np.sum(low_res_annot))
            print('organ labels to save unique = ', np.unique(low_res_annot))

            save_nifty(os.path.join(self.low_res_image_dir, fname), low_res_image)
            save_nifty(os.path.join(self.low_res_annot_dir, fname), low_res_annot)
            self.save_debug_image(fname + '_low_res', low_res_image, low_res_annot)

    def create_cropped_dataset(self):
        fnames = sorted(self.get_all_fnames())
        # if the folder doesn't exist then create it
        if not os.path.isdir(self.crop_im_dir):
            os.makedirs(self.crop_im_dir)

        if not os.path.isdir(self.crop_annot_dir):
            os.makedirs(self.crop_annot_dir)

        for fname in fnames:
            print('checking', fname, end=',')

            image_fpath = self.get_full_image_path(fname)
            annot_fpath = self.get_full_annot_path(fname)
            image_data = load_image(image_fpath)
            annot = load_annot(annot_fpath)


            image_data = np.pad(image_data, 
                                ((self.organ_padding, self.organ_padding),
                                 (self.organ_padding, self.organ_padding),
                                 (self.organ_padding, self.organ_padding)),
                                constant_values=0,
                                mode='constant')

            annot = np.pad(annot, 
                           ((self.organ_padding, self.organ_padding),
                           (self.organ_padding, self.organ_padding),
                           (self.organ_padding, self.organ_padding)),
                           constant_values=0,
                           mode='constant')
            
            (z_min, y_min, x_min,
             z_max, y_max, x_max) = im_utils.get_crop_coords_to_organ(annot, self.organ_padding)

            cropped_im = image_data[z_min:z_max,
                                    y_min:y_max,
                                    x_min:x_max]

            cropped_annot = annot[z_min:z_max,
                                  y_min:y_max,
                                  x_min:x_max]
        
            save_nifty(os.path.join(self.crop_im_dir, fname), cropped_im)
            save_nifty(os.path.join(self.crop_annot_dir, fname), cropped_annot)
            self.save_debug_image(fname + '_crop', cropped_im, cropped_annot, label_indices=[1])

