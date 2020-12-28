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
from data_prep import load_nifty

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
    plt.savefig(f'plots/image_mean_class_balance_percent.png')


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
        plt.savefig(f'plots/image_class_include_bg={include_bg}.png')

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
    plt.savefig('plots/esophagus_sizes.png')

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
    plt.savefig('plots/esophagus_percent.png')

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
    plt.savefig('plots/heart_percent.png')

if __name__ == '__main__':
    mean_class_balances()
