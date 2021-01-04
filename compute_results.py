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
from data_prep import load_nifty, save_nifty, load_im_and_heart, get_heart_centroid
from datasets import create_train_val_test_split
from skimage.transform import resize
from csv_utils import load_csv
from metrics import (get_metrics_from_arrays,
                     get_metrics_str, get_metrics,
                     get_metric_csv_row,
                     get_metric_header_str)


def best_model_name(models_dir):
    model_nums = []
    fnames = []
    for fname in os.listdir(models_dir):
        fnames.append(fname)
        num = int(fname.split('_')[0])
        model_nums.append(num)
    best_model_fname = sorted(zip(model_nums, fnames), reverse=True)[0][1]
    return best_model_fname


def get_best_loc_cnn_for_run(run):
    exp_dir = 'exp_output/struct_seg_heart_scaled_64_256'
    models_dir = os.path.join(exp_dir, 'runs', str(run), 'models')
    best_model_fname = best_model_name(models_dir)
    model_path = os.path.join(models_dir, best_model_fname) 
    print('loading model from ', model_path)
    cnn = torch.load(model_path).cuda()
    cnn = nn.DataParallel(cnn)
    cnn.eval()
    return cnn

def get_best_fine_cnn_for_run(run):
    exp_dir = 'exp_output/struct_seg_heart_cropped_fixed'
    models_dir = os.path.join(exp_dir, 'runs', str(run), 'models')
    best_model_fname = best_model_name(models_dir)
    model_path = os.path.join(models_dir, best_model_fname) 
    print('loading model from ', model_path)
    cnn = torch.load(model_path).cuda()
    cnn = nn.DataParallel(cnn)
    cnn.eval()
    return cnn

def get_best_cnn_for_run(exp_dir, run):
    models_dir = os.path.join(exp_dir, 'runs', str(run), 'models')
    best_model_fname = best_model_name(models_dir)
    model_path = os.path.join(models_dir, best_model_fname) 
    print('loading model from ', model_path)
    cnn = torch.load(model_path).cuda()
    #cnn = nn.DataParallel(cnn)
    cnn.eval()
    return cnn


def get_cropped_heart(pad_loc_seg, pad_image, patch_shape):
    # pad before extracting cropped region to ensure sufficient context
    # for a heart near the boundary.
       
    # get the central locaition of the heart
    z_mid, y_mid, x_mid = get_heart_centroid(pad_loc_seg)

    # for crop
    z_min = round(z_mid - (patch_shape[0] // 2))
    y_min = round(y_mid - (patch_shape[1] // 2))
    x_min = round(x_mid - (patch_shape[2] // 2))
    z_max = round(z_min + patch_shape[0])
    y_max = round(y_min + patch_shape[1])
    x_max = round(x_min + patch_shape[2])

    cropped_im = pad_image[z_min:z_max, y_min:y_max, x_min:x_max] 
    return cropped_im, (z_min, y_min, x_min, z_max, y_max, x_max)

def two_stage_metrics(loc_cnn, fine_cnn, data_dir, im_names):
    patch_shape = (64, 256, 256)
    image_metrics_list = []
    tps = []
    tns = []
    fps = []
    fns = []

    for im_name in im_names:
        full_res_im, full_res_annot = load_im_and_heart(join(data_dir, im_name))
        # get low res image for segmentation with loc network. 
        small_im = resize(full_res_im, patch_shape)
        # get segmentation using loc network
        loc_seg = segment(loc_cnn, small_im.astype(np.float32), 2, patch_shape, patch_shape)
        loc_seg = resize(loc_seg.astype(np.float32), full_res_im.shape, order=0)
        assert len(np.unique(loc_seg)) == 2
        print('loc seg unique = ', np.unique(loc_seg))

        # pad all to allow hearts on boundary
        pad_loc_seg = np.pad(loc_seg, 
                             ((patch_shape[0]//2, patch_shape[0]//2),
                             (patch_shape[1]//2, patch_shape[1]//2),
                             (patch_shape[2]//2, patch_shape[2]//2)),
                             constant_values = 0,
                             mode='constant')
        pad_image = np.pad(full_res_im, 
                           ((patch_shape[0]//2, patch_shape[0]//2),
                            (patch_shape[1]//2, patch_shape[1]//2),
                            (patch_shape[2]//2, patch_shape[2]//2)),
                            constant_values = 0,
                            mode='constant')
        
        cropped_heart, coords = get_cropped_heart(pad_loc_seg, pad_image,
                                                  patch_shape)
        assert cropped_heart.shape == patch_shape
        fine_seg = segment(fine_cnn, cropped_heart.astype(np.float32),
                           2, patch_shape, patch_shape)
        preds = np.zeros(pad_image.shape)
        (z_min, y_min, x_min, z_max, y_max, x_max) = coords
        preds[z_min:z_max, y_min:y_max, x_min:x_max] = fine_seg
        
        # remove padding 
        preds = preds[patch_shape[0]//2:-patch_shape[0]//2,
                      patch_shape[1]//2:-patch_shape[1]//2,
                      patch_shape[2]//2:-patch_shape[2]//2]

        foregrounds_int = full_res_annot.reshape(-1).astype(np.int)
        preds_int = preds.reshape(-1).astype(np.int)
        im_tps = (np.sum((foregrounds_int == 1) * (preds_int == 1)))
        im_tns = (np.sum((foregrounds_int == 0) * (preds_int == 0)))
        im_fps = (np.sum((foregrounds_int == 0) * (preds_int == 1)))
        im_fns = (np.sum((foregrounds_int == 1) * (preds_int == 0)))
        image_metrics = get_metrics(im_tps, im_fps, im_tns, im_fns)
        print('metrics for', im_name, get_metrics_str(image_metrics))
        image_metrics_list.append(image_metrics)
        tps.append(im_tps)
        tns.append(im_tns)
        fps.append(im_fps)
        fns.append(im_fns)

    dataset_metrics = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
    return dataset_metrics, image_metrics_list
 

def baseline_metrics(cnn, data_dir, im_names):
    patch_shape = (64, 256, 256)
    image_metrics_list = []
    tps = []
    tns = []
    fps = []
    fns = []

    for im_name in im_names:
        full_res_im, full_res_annot = load_im_and_heart(join(data_dir, im_name))
        preds = segment(cnn, full_res_im.astype(np.float32),
                        2, patch_shape, patch_shape)
        foregrounds_int = full_res_annot.reshape(-1).astype(np.int)
        preds_int = preds.reshape(-1).astype(np.int)
        im_tps = (np.sum((foregrounds_int == 1) * (preds_int == 1)))
        im_tns = (np.sum((foregrounds_int == 0) * (preds_int == 0)))
        im_fps = (np.sum((foregrounds_int == 0) * (preds_int == 1)))
        im_fns = (np.sum((foregrounds_int == 1) * (preds_int == 0)))
        image_metrics = get_metrics(im_tps, im_fps, im_tns, im_fns)
        print('metrics for', im_name, get_metrics_str(image_metrics))
        image_metrics_list.append(image_metrics)
        tps.append(im_tps)
        tns.append(im_tns)
        fps.append(im_fps)
        fns.append(im_fns)

    dataset_metrics = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
    return dataset_metrics, image_metrics_list


def compute_two_stage_metrics():
    exp_dir = 'exp_output/results_collation_01'
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    csv_path = os.path.join(exp_dir, 'two_stage.csv')
    csv_file = open(csv_path, 'w+')
    print(get_metric_header_str() + ',run,dataset', file=csv_file)
    exit()
    data_dir = 'data/ThoracicOAR'
    _, val_fnames, _ = create_train_val_test_split(data_dir)
    for run in [0, 1, 2, 3, 4]:
        loc_cnn = get_best_loc_cnn_for_run(run)
        fine_cnn = get_best_fine_cnn_for_run(run)
        start = time.time()
        metrics, im_metrics_list = two_stage_metrics(loc_cnn, fine_cnn,
                                                     data_dir, val_fnames)
        seg_dur = time.time() - start
        print(get_metric_csv_row(metrics, start) + f',{run},val', file=csv_file)
    print('csv saved to ', csv_path)
        

def compute_two_stage_test_metrics():
    exp_dir = 'exp_output/results_collation_01'
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    csv_path = os.path.join(exp_dir, 'two_stage_test.csv')
    csv_file = open(csv_path, 'w+')
    print(get_metric_header_str() + ',run,dataset', file=csv_file)
    data_dir = 'data/ThoracicOAR'
    _, _, test_fnames = create_train_val_test_split(data_dir)
    for run in [0, 1, 2, 3, 4]:
        loc_cnn = get_best_loc_cnn_for_run(run)
        fine_cnn = get_best_fine_cnn_for_run(run)
        start = time.time()
        metrics, im_metrics_list = two_stage_metrics(loc_cnn, fine_cnn,
                                                     data_dir, test_fnames)
        seg_dur = time.time() - start
        print(get_metric_csv_row(metrics, start) + f',{run},test', file=csv_file)
    print('csv saved to ', csv_path)


def compute_val_baseline_metrics():
    baseline_exp_dir = 'exp_output/struct_seg_heart_full'
    exp_dir = 'exp_output/results_collation_01'
    csv_path = os.path.join(exp_dir, 'baseline.csv')
    csv_file = open(csv_path, 'w+')
    print(get_metric_header_str() + ',run,dataset', file=csv_file)
    data_dir = 'data/ThoracicOAR'
    _, val_fnames, _ = create_train_val_test_split(data_dir)
    for run in [0, 1, 2, 3, 4, 5]:
        cnn = get_best_cnn_for_run(baseline_exp_dir, run)
        start = time.time()
        metrics, im_metrics_list = baseline_metrics(cnn, data_dir, val_fnames)
        seg_dur = time.time() - start
        print(get_metric_csv_row(metrics, start) + f',{run},val', file=csv_file)
    print('csv saved to ', csv_path)

def compute_test_baseline_metrics():
    baseline_exp_dir = 'exp_output/struct_seg_heart_full'
    exp_dir = 'exp_output/results_collation_01'
    csv_path = os.path.join(exp_dir, 'baseline_test.csv')
    csv_file = open(csv_path, 'w+')
    print(get_metric_header_str() + ',run', file=csv_file)
    data_dir = 'data/ThoracicOAR'
    _, _, test_fnames = create_train_val_test_split(data_dir)
    for run in [0, 1, 2, 3, 4, 5]:
        cnn = get_best_cnn_for_run(baseline_exp_dir, run)
        start = time.time()
        metrics, im_metrics_list = baseline_metrics(cnn, data_dir, test_fnames)
        seg_dur = time.time() - start
        print(get_metric_csv_row(metrics, start) + f',{run}', file=csv_file)
    print('csv saved to ', csv_path)


def compute_ttest():

    # Compute ttest for validation set results.
    exp_dir = 'exp_output/results_collation_01'
    baseline_csv_path = os.path.join(exp_dir, 'baseline.csv')
    baseline_dices = load_csv(baseline_csv_path, ['dice'], [float])[0]
    print('baseline dices', baseline_dices)

    two_stage_csv_path = os.path.join(exp_dir, 'two_stage.csv')
    two_stage_dices = load_csv(two_stage_csv_path, ['dice'], [float])[0]
    print('two stage dices', two_stage_dices)
    
    from scipy import stats
    result = stats.ttest_ind(baseline_dices, two_stage_dices)
    print('two stage against the baseline on validation set- ttest', result)
    
    # Compute ttest for test set results.
    baseline_csv_path = os.path.join(exp_dir, 'baseline_test.csv')
    baseline_dices = load_csv(baseline_csv_path, ['dice'], [float])[0]
    print('baseline dices', baseline_dices)

    two_stage_csv_path = os.path.join(exp_dir, 'two_stage_test.csv')
    two_stage_dices = load_csv(two_stage_csv_path, ['dice'], [float])[0]
    print('two stage dices', two_stage_dices)
    
    from scipy import stats
    result = stats.ttest_ind(baseline_dices, two_stage_dices)
    print('two stage against the baseline on test set - ttest', result)



if __name__ == '__main__':
    # compute_baseline_metrics()
    # compute_val_baseline_metrics():
    #compute_test_baseline_metrics()
    # compute_two_stage_test_metrics()
    compute_ttest()
