"""
Copyright (C) 2020-2023 Abraham George Smith

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
import argparse
import torch
import torch.nn as nn
from unet3d import UNet3D
from loss import combined_loss
import time
import numpy as np
from torch.utils.data import DataLoader
from metrics import (Metrics, get_metric_header_str)
from datasets import create_datasets
from segment_image import segment
from raw_dataset import RawDataset
from compute_results import get_converged_runs


def train_epoch(dataloader, cnn, optimizer, loss_fn):
    tps = []
    tns = []
    fps = []
    fns = []
    losses = []
    for step, (image_batch, labels_batch) in enumerate(dataloader):
        print('step', step, 'image_batch shape', image_batch.shape, 'fg in labels', torch.sum(labels_batch), end='\r')
        optimizer.zero_grad()
        outputs = cnn(image_batch.cuda())
        assert outputs.shape[2:] == image_batch.shape[2:]
        labels_batch = labels_batch.cuda().long()
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
    tps = []
    tns = []
    fps = []
    fns = []
    for _step, (image, annot) in enumerate(dataloader):
        preds = segment(cnn, image[0], 2, patch_shape)
        foregrounds_int = annot.reshape(-1).astype(int)
        preds_int = preds.reshape(-1).astype(int)
        tps.append(np.sum((foregrounds_int == 1) * (preds_int == 1)))
        tns.append(np.sum((foregrounds_int == 0) * (preds_int == 0)))
        fps.append(np.sum((foregrounds_int == 0) * (preds_int == 1)))
        fns.append(np.sum((foregrounds_int == 1) * (preds_int == 0)))
    return (np.sum(tps), np.sum(fps),
            np.sum(tns), np.sum(fns))


def log_data_split(log_dir, train_ds, val_ds, test_ds):
    # print train, val and test images to the exp_output folder for this run
    # to help check for data leakage later.
    train_split_csv = open(os.path.join(log_dir, 'train_images.txt'), 'w+')
    for image_fpath in train_ds.image_fpaths:
        print(image_fpath, file=train_split_csv)

    val_split_csv = open(os.path.join(log_dir, 'val_images.txt'), 'w+')
    for image_fpath in val_ds.image_fpaths:
        print(image_fpath, file=val_split_csv)

    test_split_csv = open(os.path.join(log_dir, 'test_images.txt'), 'w+')
    for image_fpath in test_ds.image_fpaths:
        print(image_fpath, file=test_split_csv)


def train_epochs(patience, image_fpaths, annot_fpaths, raw_ds,
                 output_dir, train_batch_size, patch_shape=None, 
                 force_fg_percent=None, organ=None):
    """ Train a network for {epochs}
        using data from {train_ds}.
        Retrun metrics at the end of each epoch.
        
        {patch_shape} is a tuple or None. If specified training will 
        be done on a random subpatch of each image with {patch_shape}
        dimensions. This enables training on images too big to
        fit in memory.
    """
    if not os.path.isdir(output_dir):
        print('Create output_dir', output_dir)
        os.makedirs(output_dir)
   
    runs_dir = os.path.join(output_dir, 'runs')
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir)
    
    # pytorch datasets
    train_ds, val_ds, test_ds = create_datasets(image_fpaths, annot_fpaths, raw_ds,
                                                patch_shape, organ, force_fg_percent)
    
    # output directory is where everything is saved
    # including logs of model performance during training.
    # and models saved during training.
    # Output dir will
    # contain subfolders numbered chronologically i.e 1,2,3,4
    total_runs = len(os.listdir(runs_dir))
    log_dir = os.path.join(runs_dir, str(total_runs))
    os.makedirs(log_dir)
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir)

    train_log_csv_path = os.path.join(log_dir, 'train_metrics.csv')
    val_log_csv_path = os.path.join(log_dir, 'val_metrics.csv')
    
    log_data_split(log_dir, train_ds, val_ds, test_ds)

    def train_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        # resize the items to the largest in the batch.
        largest_x = max([d.shape[3] for d in data])
        largest_y = max([d.shape[2] for d in data])
        largest_z = max([d.shape[1] for d in data])
        largest_x += 16-(largest_x%16)
        largest_y += 16-(largest_y%16)
        largest_z += 16-(largest_z%16)

        large_data = []
        large_target = []
        for d, t in zip(data, target):
            pad_z = max(0, largest_z - d.shape[1])
            pad_y = max(0, largest_y - d.shape[2])
            pad_x = max(0, largest_x - d.shape[3])

            pad_z_start = pad_z // 2
            pad_z_end = pad_z - pad_z_start
            pad_x_start = pad_x // 2
            pad_x_end = pad_x - pad_x_start
            pad_y_start = pad_y // 2
            pad_y_end = pad_y - pad_y_start
            d_padded = np.pad(d, ((0, 0),
                                  (pad_z_start, pad_z_end),
                                  (pad_y_start, pad_y_end),
                                  (pad_x_start, pad_x_end)))

            t_padded = np.pad(t, ((pad_z_start, pad_z_end),
                                  (pad_y_start, pad_y_end),
                                  (pad_x_start, pad_x_end)))

            large_data.append(d_padded)
            large_target.append(t_padded)

        return [torch.from_numpy(np.array(large_data)), torch.from_numpy(np.array(large_target))]

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, 
                              shuffle=True, collate_fn=train_collate,
                              num_workers=12)

    def val_collate(batch):
        # no collate required for validation.
        # first element in (im, labels)
        return batch[0]
    
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=True, num_workers=12,
                            collate_fn=val_collate)

    cnn = UNet3D(im_channels=1, out_channels=2).cuda()
    cnn = nn.DataParallel(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
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
        train_m = Metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        print(f'{train_m.csv_row(train_start)},{round(mean_loss, 4)}',
              file=train_log)
        print('epoch', epoch, 'train', train_m)
        # Validation
        cnn.eval()
        epoch_result = val_epoch(val_loader, cnn, patch_shape)
        (tps, fps, tns, fns) = epoch_result
        val_m = Metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        print(f'{val_m.csv_row(train_start)}',
              file=val_log)
        print('epoch', epoch, 'val', val_m)
        # Checkpoint and early stopping.
        if val_m.dice() > best_dice:
            epochs_without_progress = 0
            print('dice improved from', best_dice, 'to', val_m.dice())
            best_dice = val_m.dice()
            total_models = len(os.listdir(model_dir))
            new_model_path = os.path.join(
                model_dir,
                f'{total_models}_epoch_{epoch}_dice_{round(best_dice, 4)}')
            print('saving model to ', new_model_path)
            torch.save(cnn, new_model_path)
        else:
            epochs_without_progress += 1

    print(f'Training finished as {epochs_without_progress}',
          'epochs without progress')

def train_full_res(ds, until_converged, max_runs=50, force_fg_percent):
    exp_dir = f'exp_output_3090/{ds.name}_full_fg_{force_fg_percent}'
    converged, all_runs = get_converged_runs(exp_dir)
    while (len(converged) < until_converged and len(all_runs) < max_runs):
        converged, all_runs = get_converged_runs(exp_dir)
        print('training for', exp_dir)
        print(len(converged), 'converged so far')
        print(len(all_runs), 'total runs')
        train_epochs(patience=20,
                     image_fpaths=[ds.get_full_image_path(f) for f in ds.get_all_fnames()],
                     annot_fpaths=[ds.get_full_annot_path(f) for f in ds.get_all_fnames()],
                     raw_ds=ds,
                     output_dir=exp_dir,
                     force_fg_percent=force_fg_percent,
                     train_batch_size=ds.batch_size,
                     patch_shape=ds.patch_size)


def train_low_res(ds, until_converged):
    # train_localisation
    exp_dir = f'exp_output/{ds.name}_low_res'
    num_converged = len(get_converged_runs(exp_dir)[0])
    while (num_converged < until_converged):
        num_converged = len(get_converged_runs(exp_dir)[0])
        print(num_converged, 'converged so far')
        train_epochs(patience=20,
                     image_fpaths=[ds.get_low_res_image_path(f) for f in ds.get_all_fnames()],
                     annot_fpaths=[ds.get_low_res_annot_path(f) for f in ds.get_all_fnames()],
                     raw_ds=ds,
                     output_dir=exp_dir,
                     train_batch_size=ds.batch_size,
                     patch_shape=ds.patch_size)


def train_cropped(ds, until_converged):
    exp_dir = f'exp_output/{ds.name}_cropped_by_org'
    num_converged = len(get_converged_runs(exp_dir)[0])
    while (num_converged < until_converged):
        num_converged = len(get_converged_runs(exp_dir)[0])
        print(num_converged, 'converged so far')
        train_epochs(patience=20,
                     image_fpaths=[ds.get_cropped_image_path(f) for f in ds.get_all_fnames()],
                     annot_fpaths=[ds.get_cropped_annot_path(f) for f in ds.get_all_fnames()],
                     raw_ds=ds,
                     output_dir=exp_dir,
                     train_batch_size=ds.batch_size,
                     patch_shape=ds.patch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dataset')
    parser.add_argument('--organname', type=str, required=True, help='name of organ (used for exp output)')
    parser.add_argument('--imdir', type=str, required=True, help='path to full image directory')
    parser.add_argument('--annotdir', type=str, required=True, help='path to full annotation directory')
    parser.add_argument('--patchwidth', type=int, default=256,
                        help='input patch width for network (or for cropping)')
    parser.add_argument('--batchsize', type=int, default=2,
                        help='batch size for training')
    parser.add_argument('--patchdepth', type=int, default=64,
                        help='input patch depth for network (or for cropping)')
    parser.add_argument('--organpadding', type=int, default=15,
                        help=('how much to pad around the cropped organ.'
                              'Makes sure even organs on the edge of the image will have some '
                              'background context (will be padded to 0)'))
    args, _ = parser.parse_known_args()

    patch_size = (args.patchdepth, args.patchwidth, args.patchwidth)
    ds = RawDataset(args.imdir, args.annotdir, patch_size, args.organpadding,
                    batch_size=args.batchsize, name=args.organname)

    train_full_res(ds, until_converged=10, max_runs=50, force_fg_percent=80)
    train_cropped(ds, until_converged=10)
    train_low_res(ds, until_converged=10)
