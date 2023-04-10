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
        total_scan = len(organ_labels.reshape(-1))
        percent_organ = (total_organ / total_scan)
        total_fg += total_organ
        total_vox += total_scan
    print(f'{ds.name}_fg_percent =', round(((total_fg/total_vox) * 100)))


def mean_class_balances_decath():
    im_sizes = []
    names = ['spleen', 'pancreas', 'left atrium', 'prostate', 'liver']
    percents = [0.43780517578125,
                0.21805809823654426, 
                0.4020698308839718,
                2.7093768054989127,
                2.2849260059115855]

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
