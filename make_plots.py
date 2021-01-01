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
import numpy as np
import nibabel as nib

from skimage.transform import resize
from skimage import img_as_float
from skimage.io import imsave
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import load_im_and_heart
from csv_utils import load_csv

def heart_percent_plot():
    heart_percents = []
    im_dir_names = os.listdir(input_dir)
    for im_name in im_dir_names:
        full_im_data_dir = os.path.join(input_dir, im_name)
        image_data, heart_labels = load_im_and_heart(full_im_data_dir)
        total_heart = np.sum(heart_labels > 0)
        total_scan = len(heart_labels.reshape(-1))
        percent_heart = (total_heart / total_scan) * 100
        heart_percents.append(percent_heart)
    x = range(len(heart_percents))
    y = heart_percents
    plt.title('Percentage of total image labeled heart')
    plt.bar(x, y)
    plt.xlabel('Image')
    plt.ylabel('Heart Percent')
    plt.savefig('plots/heart_percent.png')


def load_train_dice():
    fname = 'train_metrics_combined_quarter_1.csv'
    seconds, dices = load_csv('logs/' + fname,
                              ['seconds', 'dice'],
                              [float, float])
    return seconds[:200], dices[:200]


def plot_dice(log_dir):
    # First plot dice over epochs for both training and validation.
    for x_axis in ['time', 'epochs']:
        plt.figure(figsize=(16, 9))
        clrs = sns.color_palette("husl", 5)
        with sns.axes_style('white'):
            plt.grid()
            plt.yticks(np.arange(0, 1.025, 0.025))
            runs_dir = os.path.join(log_dir, 'runs')
            for run_dir in os.listdir(runs_dir):
                train_csv = os.path.join(runs_dir, run_dir, 'train_metrics.csv')
                seconds, dices = load_csv(train_csv,
                                          ['seconds', 'dice'],
                                          [float, float])
                minutes = [s/60 for s in seconds]
                dices = np.array(dices)
                dices[np.isnan(dices)] = 0
                epochs = range(len(dices))
                #plt.yticks(np.arange(0, 36, 2))
                p = None
                plt.ylabel('dice')
                if x_axis == 'time':
                    plt.xlabel('minutes')
                    p = plt.plot(minutes, dices,
                                 label=f'training dice {run_dir}, max: {round(max(dices), 3)}')
                else:
                    plt.xlabel('epochs')
                    p = plt.plot(epochs, dices,
                                 label=f'training dice {run_dir}, max: {round(max(dices), 3)}')

                prev_color = p[0].get_color()
                val_csv = os.path.join(runs_dir, run_dir, 'val_metrics.csv')
                seconds, dices = load_csv(val_csv,
                                          ['seconds', 'dice'],
                                          [float, float])
                dices = np.array(dices)
                dices[np.isnan(dices)] = 0
                epochs = range(len(dices))
                #plt.yticks(np.arange(0, 36, 2))
                p = None
                if x_axis == 'time':
                    plt.plot(minutes, dices, color=prev_color,
                             label=f'validation dice {run_dir}, max: {round(max(dices), 3)}',
                             linestyle='--')
                else:
                    plt.plot(epochs, dices, color=prev_color,
                             label=f'validation dice {run_dir}, max: {round(max(dices), 3)}',
                             linestyle='--')

            # plt.fill_between(idx, means-stds, means+stds,
            #                  alpha=0.3, facecolor=clrs[0])
            plt.legend()

        if not os.path.isdir(os.path.join(log_dir, 'plots')):
            os.makedirs(os.path.join(log_dir, 'plots'))

        fig_path = os.path.join(log_dir, 'plots', 'train_val_dice_' + x_axis + '.png')
        print('saving figure to ', fig_path)
        plt.savefig(fig_path)
        # Then plot dice over time (minutes) for both training and validation.


def show_central_heart_slice(input_dir, im_dir_name = '1'):
    im_data_dir = os.path.join(input_dir, im_dir_name)
    assert os.path.isdir(im_data_dir), f'{im_data_dir} required. Did you download struct seg?'
    # get data and heart labels for a patch with the heart in it
    image_data, heart_labels = load_im_and_heart(im_data_dir)
    print('heart labels sum = ', np.sum(heart_labels))
    print('im shape', image_data.shape)
    print('heart shape', heart_labels.shape)
    for i in range(heart_labels.shape[0]):
        slice_im = image_data[i]
        slice_labels = heart_labels[i]
        im_for_show = np.hstack((slice_im, slice_labels*np.max(slice_im)))
        imsave(f'slices/central_slice_{i}.png', im_for_show)

if __name__ == '__main__':
    # output_dir = os.path.join('data', 'ThoracicOAR_eighth')
    # show_central_heart_slice(output_dir, im_dir_name = '1')
    # plot_dice('train_output/struct_seg_heart_quarter_adam_less_cx')
    #plot_dice('train_output/struct_seg_heart_full')
    plot_dice('train_output/struct_seg_heart_quarter_lr_1e-4')
    #plot_dice('train_output/struct_seg_heart_quarter_30')
