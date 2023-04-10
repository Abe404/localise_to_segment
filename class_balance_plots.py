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

from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import texfig
import argparse
import im_utils
from raw_dataset import RawDataset

def autolabel_percent(bars):
    # attach a text label to each rect in bars
    for rect in bars:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 4), # 4 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


def mean_class_balances():
    raise Exception('old code for struct seg - using decathlon now')
    #plt.figure(figsize=(12*0.8, 8*0.8))
    #plt.title('Organ Class ')
    im_sizes = []
    class_names = ['background', 'left lung', 'right lung', 'heart',
                   'esophagus', 'trachia', 'spine']
    annot_sizes = []
    class_percents = [[] for _ in class_names]
    for i in range(1, 51):
        data_dir = os.path.join('data', 'ThoracicOAR', str(i))
        annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))
        annot_sizes.append(annot.size)
        for i, name in enumerate(class_names):
            organ_size = np.sum(annot==i)
            class_percent = 100 * (organ_size / annot.size)
            class_percents[i].append(class_percent)
   
    # ax = plt.axes()
    # ax.yaxis.grid()

    mean_class_percents = [] 
    for i, name in enumerate(class_names):
        mean_class_percents.append(np.mean(class_percents[i]))
    
    names = class_names[1:]
    mean_class_percents = mean_class_percents[1:]

    rects = plt.bar(list(range(len(mean_class_percents))), mean_class_percents,
                    color=(plt.rcParams['axes.prop_cycle'].by_key()['color']))
    autolabel_percent(rects)
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.ylim([0, 1.2])
    plt.ylabel('Percentage of total image')
    plt.xlabel('Organ')
    plt.xticks(range(len(names)), names)
    plt.savefig(f'png_plots/image_mean_class_balance_percent.png')
    texfig.savefig(f'latex_plots/image_mean_class_balance_percent')


def image_1_class_balance():
    for include_bg in [True, False]:
        plt.figure(figsize=(12*0.8, 8*0.8))
        plt.title('Image 1 class balance')
        data_dir = os.path.join('data', 'ThoracicOAR', '1')
        annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))

        class_names = ['background', 'lung1', 'lung2', 'heart',
                       'esophagus', 'trachia', 'spine']

        names = []
        vals = []
        for i, name in enumerate(class_names):

            if name != 'background' or include_bg:
                names.append(name)
                vals.append(np.sum(annot==i))


        rects = plt.bar(list(range(len(vals))), vals,
                        color=(plt.rcParams['axes.prop_cycle'].by_key()['color']))

        autolabel(rects)
        plt.ylabel('number of voxels')
        plt.xlabel('class')
        plt.xticks(range(len(names)), names)
        plt.savefig(f'png_plots/image_class_include_bg={include_bg}.png')

def plot_esophagus_sizes():
    plt.figure(figsize=(12*0.8, 8*0.8))
    plt.ylabel('number of esophagus voxels')
    plt.xlabel('Image (ordered by esophagus size)')
    vals = []
    names = []
    for i in range(1, 51):
        data_dir = os.path.join('data', 'ThoracicOAR', str(i))
        annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))
        vals.append(np.sum(annot==4))
        names.append(str(i))
    plt.grid()
    vals, names = zip(*sorted(zip(vals, names)))
    rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][4]] * 50))
    #autolabel(rects)
    #plt.xticks(range(len(names)), names)
    plt.savefig('png_plots/esophagus_sizes.png')

    ### as %
    plt.figure(figsize=(12*0.8, 8*0.8))
    plt.ylabel('Esophagus % of image')
    plt.xlabel('Image (ordered by esophagus size)')
    vals = []
    names = []
    for i in range(1, 51):
        data_dir = os.path.join('data', 'ThoracicOAR', str(i))
        annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))
        vals.append(100 * ((np.sum(annot==4)) / annot.size))
        names.append(str(i))

    plt.grid()
    vals, names = zip(*sorted(zip(vals, names)))
    rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][4]] * 50))
    #autolabel(rects)
    #plt.xticks(range(len(names)), names)
    plt.savefig('png_plots/esophagus_percent.png')

def plot_heart_percent():
    ### Heart % of image.
    plt.figure(figsize=(12*0.8, 8*0.8))
    plt.ylabel('Heart % of image')
    plt.xlabel('Image (ordered by heart size)')
    vals = []
    names = []
    for i in range(1, 51):
        data_dir = os.path.join('data', 'ThoracicOAR', str(i))
        annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))
        vals.append(100 * ((np.sum(annot==3)) / annot.size))
        names.append(str(i))
    plt.grid()
    vals, names = zip(*sorted(zip(vals, names)))
    rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][3]] * 50))
    #autolabel(rects)
    #plt.xticks(range(len(names)), names)
    plt.savefig('png_plots/heart_percent.png')


def decathlon_class_balance(ds):
    # load the decathlon datasets.
    # get total bg and total fg for each dataset.

    fnames = sorted(ds.get_all_fnames())
    total_fg = 0
    total_vox = 0
    for fname in fnames:
        annot_fpath = ds.get_full_annot_path(fname)
        organ_labels = im_utils.load_annot(annot_fpath)
        total_organ = np.sum(organ_labels)
        #print('organ unique', np.unique(organ_labels))
        total_scan = len(organ_labels.reshape(-1))
        #print('total_scan', total_scan)
        percent_organ = (total_organ / total_scan)
        #print('Loading image,',
        #      'actual', total_organ,
        #      'out of total', total_scan,
        #       'giving percent', 100 * percent_organ)
        total_fg += total_organ
        total_vox += total_scan
    print(f'{ds.name}_fg_percent =', round(((total_fg/total_vox) * 100)))
    #print('total_vox = ', total_vox)
    #print('total_fg = ', total_fg)



def mean_class_balances_decath():
    #plt.figure(figsize=(12*0.8, 8*0.8))
    #plt.title('Organ Class ')
    im_sizes = []
    names = ['spleen', 'pancreas', 'left atrium', 'prostate', 'liver']
    percents = [0.43780517578125,
                0.21805809823654426, 
                0.4020698308839718,
                2.7093768054989127,
                2.2849260059115855]

    # ax = plt.axes()
    # ax.yaxis.grid()

    rects = plt.bar(list(range(len(percents))), percents,
                    color=(plt.rcParams['axes.prop_cycle'].by_key()['color']))
    autolabel_percent(rects)
    plt.yticks(np.arange(0, 3.0, 0.25))
    plt.ylim([0, 3.0])
    plt.ylabel('Percentage of total image')
    plt.xlabel('Organ')
    plt.xticks(range(len(names)), names)
    plt.savefig(f'png_plots/image_mean_class_balance_percent.png')
    texfig.savefig(f'latex_plots/image_mean_class_balance_percent')


if __name__ == '__main__':

    mean_class_balances_decath()
    
    if False:
        parser = argparse.ArgumentParser(description='Dataset stats')
        parser.add_argument('--organname', type=str, required=True, help='name of organ')
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
        decathlon_class_balance(ds)




