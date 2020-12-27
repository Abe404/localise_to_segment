import os
import numpy as np
import nibabel as nib

from skimage.transform import resize
from skimage import img_as_float
from skimage.io import imsave
import shutil
import matplotlib.pyplot as plt

from data_prep import load_im_and_heart

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
    output_dir = os.path.join('data', 'ThoracicOAR_eighth')
    show_central_heart_slice(output_dir, im_dir_name = '1')
