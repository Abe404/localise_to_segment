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
import torch.nn.functional as F
import torch
import torch.nn as nn
from unet3d import UNet3D
from loss import combined_loss
import time
import numpy as np
from torch.utils.data import DataLoader
from metrics import (get_metrics_from_arrays,
                     get_metrics_str, get_metrics,
                     get_metric_csv_row,
                     get_metric_header_str)
from datasets import create_datasets
from segment_image import segment


def train_epoch(dataloader, cnn, optimizer, loss_fn):
    start = time.time()
    tps = []
    tns = []
    fps = []
    fns = []
    losses = []
    for step, (image_batch, labels_batch) in enumerate(dataloader):
        print('step', step, end='\r')
        optimizer.zero_grad()
        outputs = cnn(image_batch.cuda())
        labels_batch  = labels_batch.cuda().long()
        loss = loss_fn(outputs, labels_batch)
        losses.append(loss.cpu().item())
        preds_int = torch.argmax(outputs, 1)
        foregrounds_int = labels_batch
        tps.append(torch.sum((foregrounds_int == 1) *
                             (preds_int == 1)).cpu().numpy())
        tns.append(torch.sum((foregrounds_int == 0) *
                             (preds_int == 0)).cpu().numpy())
        fps.append(torch.sum((foregrounds_int == 0) *
                             (preds_int == 1)).cpu().numpy())
        fns.append(torch.sum((foregrounds_int == 1) *
                             (preds_int == 0)).cpu().numpy())
        loss.backward()
        optimizer.step()
    
    return (np.sum(tps), np.sum(fps),
        np.sum(tns), np.sum(fns),
        np.mean(losses))


def val_epoch(dataloader, cnn, patch_shape):
    start = time.time()
    tps = []
    tns = []
    fps = []
    fns = []
    for step, (image, annot) in enumerate(dataloader):
        preds = segment(cnn, image[0], 2, patch_shape, patch_shape)
        foregrounds_int = annot.reshape(-1).astype(np.int)
        preds_int = preds.reshape(-1).astype(np.int)
        tps.append(np.sum((foregrounds_int == 1) * (preds_int == 1)))
        tns.append(np.sum((foregrounds_int == 0) * (preds_int == 0)))
        fps.append(np.sum((foregrounds_int == 0) * (preds_int == 1)))
        fns.append(np.sum((foregrounds_int == 1) * (preds_int == 0)))
    return (np.sum(tps), np.sum(fps),
            np.sum(tns), np.sum(fns))


def train_epoch(dataloader, cnn, optimizer, loss_fn):
    start = time.time()
    tps = []
    tns = []
    fps = []
    fns = []
    losses = []
    for step, (image_batch, labels_batch) in enumerate(dataloader):
        print('step', step, 'batch_shape', image_batch.shape, end='\r')
        optimizer.zero_grad()
        outputs = cnn(image_batch.cuda())
        labels_batch  = labels_batch.cuda().long()
        loss = loss_fn(outputs, labels_batch)
        losses.append(loss.cpu().item())
        preds_int = torch.argmax(outputs, 1)
        foregrounds_int = labels_batch
        tps.append(torch.sum((foregrounds_int == 1) *
                             (preds_int == 1)).cpu().numpy())
        tns.append(torch.sum((foregrounds_int == 0) *
                             (preds_int == 0)).cpu().numpy())
        fps.append(torch.sum((foregrounds_int == 0) *
                             (preds_int == 1)).cpu().numpy())
        fns.append(torch.sum((foregrounds_int == 1) *
                             (preds_int == 0)).cpu().numpy())
        loss.backward()
        optimizer.step()
    
    return (np.sum(tps), np.sum(fps),
        np.sum(tns), np.sum(fns),
        np.mean(losses))


def train_epochs(patience, data_dir, output_dir, train_batch_size, patch_shape=None):
    """ Train a network for {epochs}
        using data from {train_ds}.
        Retrun metrics at the end of each epoch.
        
        {patch_size} is a tuple or None. If specified training will 
        be done on a random subpatch of each image with {patch_size}
        dimensions. This enables training on images too big to
        fit in memory.
    """
    if not os.path.isdir(output_dir):
        print('Create output_dir', output_dir)
        os.makedirs(output_dir)
   
    runs_dir = os.path.join(output_dir, 'runs')
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir)


    train_ds, val_ds, test_ds = create_datasets(data_dir, patch_shape)
    
    # output directory is where everything is saved
    # including logs of model performance during training.
    # and models saved during training.
    # We assume there will be some reporition. So the output dir will
    # contain subfolders numbered chronologically i.e 1,2,3,4
    total_runs = len(os.listdir(runs_dir))
    log_dir = os.path.join(runs_dir, str(total_runs))
    os.makedirs(log_dir)
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir)

    train_log_csv_path = os.path.join(log_dir, 'train_metrics.csv')
    val_log_csv_path = os.path.join(log_dir, 'val_metrics.csv')
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, 
                              shuffle=True, num_workers=12)

    def val_collate(batch):
        # no collate required for validation.
        # first element in (im, labels)
        return batch[0]
    
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=True, num_workers=12,
                            collate_fn=val_collate)
    cnn = UNet3D(im_channels=1, out_channels=2).cuda()
    cnn = nn.DataParallel(cnn)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(cnn.parameters())
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01,
    #                            momentum=0.99, nesterov=True)
    loss_fn = combined_loss
    train_log = open(train_log_csv_path, 'w+')
    val_log = open(val_log_csv_path, 'w+')
    print(get_metric_header_str() + ',loss', file=train_log)
    print(get_metric_header_str(), file=val_log)
    best_dice = 0
    epochs_without_progress = 0
    epoch = 0
    train_start = time.time()
    while epochs_without_progress < patience:
        epoch += 1
        # Train
        cnn.train()
        epoch_result = train_epoch(train_loader, cnn, optimizer, combined_loss)
        (tps, fps, tns, fns, mean_loss) = epoch_result
        train_m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        print(f'{get_metric_csv_row(train_m, train_start)},{round(mean_loss, 4)}',
              file=train_log)
        print('epoch', epoch, 'train', get_metrics_str(train_m))
        # Validation
        cnn.eval()
        epoch_result = val_epoch(val_loader, cnn, patch_shape)
        (tps, fps, tns, fns) = epoch_result
        val_m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        print(f'{get_metric_csv_row(val_m, train_start)}',
              file=val_log)
        print('epoch', epoch, 'val', get_metrics_str(val_m))
        # Checkpoint and early stopping.
        if val_m['dice'] > best_dice:
            epochs_without_progress = 0
            print('dice improved from', best_dice, 'to', val_m['dice'])
            best_dice = val_m['dice']
            total_models = len(os.listdir(model_dir))
            new_model_path = os.path.join(model_dir,
                f'{total_models}_epoch_{epoch}_dice_{round(best_dice, 4)}')
            print('saving model to ', new_model_path)
            torch.save(cnn, new_model_path)
        else:
            epochs_without_progress += 1

    print(f'Training finished as {epochs_without_progress}',
           'epochs without progress')

def train_full_res():
    train_epochs(patience=20,
                 data_dir=os.path.join('data', 'ThoracicOAR'),
                 output_dir='exp_output/struct_seg_heart_full',
                 train_batch_size=2,
                 patch_shape=(64,256,256))

def train_quarter_res():
    train_epochs(patience=20,
                 data_dir=os.path.join('data', 'ThoracicOAR_quarter'),
                 output_dir='exp_output/struct_seg_heart_quarter_30',
                 train_batch_size=10,
                 patch_shape=(48,128,128)) # Full image size. No random cropping.

def train_64_256_res():
    train_epochs(patience=20,
                 data_dir=os.path.join('data', 'ThoracicOAR_scaled_64_256'),
                 output_dir='exp_output/struct_seg_heart_scaled_64_256',
                 train_batch_size=2,
                 patch_shape=(64,256,256)) # Full image size. No random cropping.


def train_cropped():
    train_epochs(patience=20,
                 data_dir=os.path.join('data', 'ThoracicOAR_cropped'),
                 output_dir='exp_output/struct_seg_heart_cropped_fixed',
                 train_batch_size=2,
                 patch_shape=(64,256,256)) # Full image size. No random cropping.

def train_cropped_by_network():
    train_epochs(patience=20,
                 data_dir=os.path.join('data', 'ThoracicOAR_cropped_using_network'),
                 output_dir='exp_output/struct_seg_heart_cropped_by_network',
                 train_batch_size=2,
                 patch_shape=(64,256,256)) # Full image size. No random cropping.


if __name__ == '__main__':
    for i in range(6):
        train_64_256_res()
