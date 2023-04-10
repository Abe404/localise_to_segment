"""
Copyright (C) 2021,2022,2023 Abraham George Smith

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
import im_utils
import torch
import time
import numpy as np
from scipy import stats
from skimage.transform import resize
import im_utils
from segment_image import segment
from csv_utils import load_csv
from raw_dataset import RawDataset
from metrics import Metrics, metric_headers, compute_metrics_from_binary_masks

def best_model_name(models_dir):
    model_nums = []
    fnames = []
    for fname in os.listdir(models_dir):
        fnames.append(fname)
        num = int(fname.split('_')[0])
        model_nums.append(num)
    best_model_fname = sorted(zip(model_nums, fnames), reverse=True)[0][1]
    return best_model_fname

def get_best_cnn_for_run(exp_dir, run):
    models_dir = os.path.join(exp_dir, 'runs', str(run), 'models')
    best_model_fname = best_model_name(models_dir)
    model_path = os.path.join(models_dir, best_model_fname) 
    print('loading model from ', model_path)
    cnn = torch.load(model_path).cuda()
    #cnn = nn.DataParallel(cnn)
    cnn.eval()
    return cnn


def two_stage_metrics(loc_cnn, fine_cnn, im_names, raw_ds, gt_loc=False):
    """
        loc_cnn - Trained localisation network model
        fine_cnn - Trained organ segmentation model
        im_names - File names to segment
        gt_loc - use labels to localise (artificially improves accuracy)


        returns:
             - dataset_metrics - Metrics summary for entire dataset
             - image_metrics_list - Metrics for each individual image
    """ 
    image_metrics_list = []
    dataset_metrics = Metrics() 

    for fname in im_names:

        full_im_fpath = raw_ds.get_full_image_path(fname)
        full_im = im_utils.load_image(full_im_fpath)
        full_annot_fpath = raw_ds.get_full_annot_path(fname)
        full_annot = im_utils.load_annot(full_annot_fpath)

        if gt_loc:
            # if using ground truth to localise, then we do not need to segment the low res image
            # we can just load the full res annotation and return as a segmentation before padding. 
            loc_seg = full_annot  # segmentation used for localisation
            print('using gt loc')
        else:
            # load the low_res image and segment with the loc_cnn
            low_res_im_fpath = raw_ds.get_low_res_image_path(fname)
            low_res_im = im_utils.load_image(low_res_im_fpath)
            # get segmentation using loc network
            loc_seg = segment(loc_cnn, low_res_im.astype(np.float32),
                              raw_ds.batch_size, raw_ds.patch_size)
            # resize loc_seg to full image to get coordinates relative to the full image
            loc_seg = resize(loc_seg.astype(np.float32), full_im.shape, order=0)
            assert len(np.unique(loc_seg)) == 2, "should be just 0 and 1 for loc_seg"
        
    
        # pad by half patch size to ensure even organs on the edge have surrounding context
        # and they will still appear closer to the center of the patch.
        pad_loc_seg = np.pad(loc_seg, 
                             ((raw_ds.patch_size[0]//2, raw_ds.patch_size[0]//2),
                             (raw_ds.patch_size[1]//2, raw_ds.patch_size[1]//2),
                             (raw_ds.patch_size[2]//2, raw_ds.patch_size[2]//2)),
                             constant_values = 0,
                             mode='constant')

        pad_image = np.pad(full_im, 
                           ((raw_ds.patch_size[0]//2, raw_ds.patch_size[0]//2),
                            (raw_ds.patch_size[1]//2, raw_ds.patch_size[1]//2),
                            (raw_ds.patch_size[2]//2, raw_ds.patch_size[2]//2)),
                            constant_values = 0,
                            mode='constant')

        # get bounds of the organ, with some extra padding included
        (z_min, y_min, x_min,
         z_max, y_max, x_max) = im_utils.get_crop_coords_to_organ(pad_loc_seg, raw_ds.organ_padding)

        cropped_im = pad_image[z_min:z_max,
                               y_min:y_max,
                               x_min:x_max]

        # organ segmentation using cropped region of image
        fine_seg = segment(fine_cnn, cropped_im.astype(np.float32),
                           raw_ds.batch_size, raw_ds.patch_size)


        preds = np.zeros(pad_image.shape) # init empty array to store predictions
        preds[z_min:z_max, y_min:y_max, x_min:x_max] = fine_seg # assign segmented region
        
        # remove padding that was added to all edges of image before cropping.
        preds = preds[raw_ds.patch_size[0]//2:-raw_ds.patch_size[0]//2,
                      raw_ds.patch_size[1]//2:-raw_ds.patch_size[1]//2,
                      raw_ds.patch_size[2]//2:-raw_ds.patch_size[2]//2]
        
        # get ground truth fg to compute metrics
        foregrounds_int = full_annot.reshape(-1).astype(int)
        preds_int = preds.reshape(-1).astype(int)
        im_metrics = compute_metrics_from_binary_masks(seg=preds_int, gt=foregrounds_int)
        print('metrics for', fname, im_metrics)
        image_metrics_list.append(im_metrics)
        print('dataset_metrics ', dataset_metrics)
        print('im_metrics ', im_metrics)
        dataset_metrics = dataset_metrics + im_metrics # we compute all tps etc for the dataset

    return dataset_metrics, image_metrics_list
 

def baseline_metrics(cnn, im_names, raw_ds):
    image_metrics_list = []
    dataset_metrics = Metrics()
    for fname in im_names:
        full_im_fpath = raw_ds.get_full_image_path(fname)
        full_im = im_utils.load_image(full_im_fpath)
        full_annot_fpath = raw_ds.get_full_annot_path(fname)
        full_annot = im_utils.load_annot(full_annot_fpath)
        preds = segment(cnn, full_im.astype(np.float32), raw_ds.batch_size, raw_ds.patch_size)
        im_metrics = compute_metrics_from_binary_masks(seg=preds, gt=full_annot)
        print('metrics for', fname, im_metrics)
        image_metrics_list.append(im_metrics)
        dataset_metrics = dataset_metrics + im_metrics
    return dataset_metrics, image_metrics_list


def get_converged_runs(exp_dir):
    """ get list of all the runs
        and then return those that obtained aa dice score above 0.1 after 10 epochs
    """
    runs_dir = os.path.join(exp_dir, 'runs')
    if not os.path.isdir(runs_dir):
        return [], []
    all_runs = sorted(os.listdir(runs_dir))
    converged_runs = []
    for run in all_runs:
        run_dir = os.path.join(runs_dir, str(run))
        val_csv_path = os.path.join(run_dir, 'val_metrics.csv')
        dices = load_csv(val_csv_path, ['dice'], [float])[0]
        # filter out NaN values. 
        # max will return NaN if any value are NaN. This led to an
        # error where a run would be detected as not converged if any values were NaN
        # even if the dice eventually got higher than 0.1
        dices = [d for d in dices if not np.isnan(d)] 
        if dices and max(dices) > 0.1:
            converged_runs.append(run)
    return converged_runs, all_runs

    
def compute_two_stage_metrics(loc_exp_dir, organ_exp_dir, im_names,
                              csv_path, raw_ds, gt_loc):
    csv_file = open(csv_path, 'w+')


    print(','.join(metric_headers) + ',loc_run,organ_run', file=csv_file)

    loc_converged_runs, loc_all_runs = get_converged_runs(loc_exp_dir)
    print('converged loc runs = ', len(loc_converged_runs),
          'out of', len(loc_all_runs))

    org_converged_runs, org_all_runs = get_converged_runs(organ_exp_dir)
    print('converged org runs = ', len(org_converged_runs),
          'out of', len(org_all_runs))

    # restrict the lists to the length of the shortest so there is a match for each
    shortest_list_len = min(len(loc_converged_runs), len(org_converged_runs))
    loc_converged_runs = loc_converged_runs[:shortest_list_len]
    org_converged_runs = org_converged_runs[:shortest_list_len]
    print(f'using only {shortest_list_len} matching runs')

    assert len(loc_converged_runs) == len(org_converged_runs), (
        " converged runs for each network should be the same so we can use them"
        " together and compute metrics for each two-stage combination")
        
    for loc_run, organ_run in zip(loc_converged_runs, org_converged_runs):
        loc_cnn = get_best_cnn_for_run(loc_exp_dir, loc_run)
        fine_cnn = get_best_cnn_for_run(organ_exp_dir, organ_run)
        start = time.time()
        metrics, _im_metrics_list = two_stage_metrics(loc_cnn, fine_cnn, im_names, raw_ds, gt_loc)
        print(metrics.csv_row(start) + f',{loc_run},{organ_run}', file=csv_file)
    print('csv saved to ', csv_path)


def compute_baseline_metrics(baseline_exp_dir, im_names, raw_ds, csv_path):
    csv_file = open(csv_path, 'w+')

    print(','.join(metric_headers) + ',run', file=csv_file)
    converged_runs, all_runs = get_converged_runs(baseline_exp_dir)
    print('converged baseline runs = ', len(converged_runs), 'out of', len(all_runs))
    for run in converged_runs:

        cnn = get_best_cnn_for_run(baseline_exp_dir, run)
        start = time.time()
        metrics, _im_metrics_list = baseline_metrics(cnn, im_names, raw_ds)
        print(metrics.csv_row(start) + f',{run}', file=csv_file)
    print('csv saved to ', csv_path)

def compute_ttest(results_dir, organ, dataset):
    # Compute ttest for results.
    baseline_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_baseline.csv')
    baseline_dices = load_csv(baseline_csv_path, ['dice'], [float])[0]
    two_stage_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_results_two_stage.csv')
    two_stage_dices = load_csv(two_stage_csv_path, ['dice'], [float])[0]

    # baseline vs two stage (most important results)
    result = stats.ttest_ind(baseline_dices, two_stage_dices)
    print(organ, f'two stage against the baseline on {dataset} set - ttest', result)

    # baseline vs gt loc 
    gt_loc_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_results_two_stage_gt_loc.csv')
    gt_loc_dices = load_csv(gt_loc_csv_path, ['dice'], [float])[0]

    result = stats.ttest_ind(baseline_dices, gt_loc_dices)
    print(organ, f'gt loc against the baseline on {dataset} set - ttest', result)

    # two stage vs gt loc 
    result = stats.ttest_ind(two_stage_dices, gt_loc_dices)
    print(organ, f'gt loc against the two stage{dataset} set - ttest', result)

    
def log_means(organ, dataset):
    # Compute ttest for validation set results.
    results_dir = 'results_3090'
    output_csv = open(os.path.join(results_dir, f'{organ}_{dataset}_means.csv'), 'w+')
    print('method,mean,std', file=output_csv)
    baseline_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_baseline.csv')
    baseline_dices = load_csv(baseline_csv_path, ['dice'], [float])[0]
    baseline_dice_mean = round(np.mean(baseline_dices), 4)
    baseline_dice_std = round(np.std(baseline_dices), 4)
    print(f'baseline,{baseline_dice_mean},{baseline_dice_std}', file=output_csv)
    two_stage_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_results_two_stage.csv')
    two_stage_dices = load_csv(two_stage_csv_path, ['dice'], [float])[0]
    two_stage_dice_mean = round(np.mean(two_stage_dices), 4)
    two_stage_dice_std = round(np.std(two_stage_dices), 4)
    print(f'two_stage,{two_stage_dice_mean},{two_stage_dice_std}', file=output_csv)

    two_stage_gt_loc_csv_path = os.path.join(results_dir, f'{organ}_{dataset}_results_two_stage_gt_loc.csv')
    two_stage_gt_loc_dices = load_csv(two_stage_gt_loc_csv_path, ['dice'], [float])[0]
    two_stage_gt_loc_dice_mean = round(np.mean(two_stage_gt_loc_dices), 4)
    two_stage_gt_loc_dice_std = round(np.std(two_stage_gt_loc_dices), 4)
    print(f'two_stage_gt_loc,{two_stage_gt_loc_dice_mean},{two_stage_gt_loc_dice_std}', file=output_csv)

    print(f'{organ}_two_stage_minus_baseline_dice_mean =', two_stage_dice_mean - baseline_dice_mean)
    print(f'{organ}_two_stage_gt_loc_minus_baseline_dice_mean =', two_stage_gt_loc_dice_mean - baseline_dice_mean)


def compute_metrics(im_dir, annot_dir, organ_padding,
                    patch_width, patch_depth, batch_size, organ, dataset):

    patch_size = (patch_depth, patch_width, patch_width)
    ds = RawDataset(im_dir, annot_dir, patch_size, organ_padding, batch_size=batch_size)
    exp_output_dir = 'exp_output_3090'
    results_dir = 'results_3090'
    exp_modes = ['low_res', 'full', 'cropped_by_org']
    # load the {dataset} set from the runs output
    fpaths = open(f'{exp_output_dir}/{organ}_{exp_modes[0]}/runs/0/{dataset}_images.txt').readlines()
    im_names = [os.path.basename(f).strip() for f in fpaths]
    
    # and also check that the {dataset} set is the same for all runs for all exp_modes
    for exp_mode in exp_modes:
        runs_dir = f'{exp_output_dir}/{organ}_{exp_mode}/runs/'
        num_runs = len(os.listdir(runs_dir))
        for i in range(num_runs):
            dataset_split_fpath = os.path.join(runs_dir, str(i), f'{dataset}_images.txt')
            run_dataset_fpaths = open(dataset_split_fpath).readlines()
            dataset_im_names = [os.path.basename(f).strip() for f in run_dataset_fpaths] 
            assert sorted(im_names) == sorted(im_names), (
                f"run dir {i} does not have same {dataset} images as run 0"
                f"run {i} {dataset} images are {dataset_im_names} and the run 0"
                f"{dataset} images are {im_names}")

    compute_two_stage_metrics(loc_exp_dir=f'{exp_output_dir}/{organ}_low_res', 
                              organ_exp_dir=f'{exp_output_dir}/{organ}_cropped_by_org', 
                              im_names=im_names,
                              csv_path=f'{results_dir}/{organ}_{dataset}_results_two_stage.csv', 
                              raw_ds=ds,
                              gt_loc=False)

    compute_two_stage_metrics(loc_exp_dir=f'{exp_output_dir}/{organ}_low_res', 
                              organ_exp_dir=f'{exp_output_dir}/{organ}_cropped_by_org', 
                              im_names=im_names,
                              csv_path=f'{results_dir}/{organ}_{dataset}_results_two_stage_gt_loc.csv', 
                              raw_ds=ds,
                              gt_loc=True)

    compute_baseline_metrics(baseline_exp_dir=f'{exp_output_dir}/{organ}_full',
                             im_names=im_names, 
                             raw_ds=ds,
                             csv_path=f'{results_dir}/{organ}_{dataset}_baseline.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--imdir', type=str, required=True, help='path to full image directory')
    parser.add_argument('--annotdir', type=str, required=True, help='path to full annotation directory')
    parser.add_argument('--patchwidth', type=int, default=256,
                        help='input patch width for network (or for cropping)')
    parser.add_argument('--patchdepth', type=int, default=64,
                        help='input patch depth for network (or for cropping)')
    parser.add_argument('--organpadding', type=int, default=15,
                        help=('how much to pad around the cropped organ.'
                              'Makes sure even organs on the edge of the image will have some '
                              'background context (will be padded to 0)'))
    parser.add_argument('--organname', type=str)
    parser.add_argument('--dataset', type=str) # val or test
    args, _ = parser.parse_known_args()
    compute_metrics(args.imdir, args.annotdir, args.organpadding,
                    args.patchwidth, args.patchdepth, batch_size=2,
                    organ=args.organname, dataset=args.dataset)
    log_means(organ=args.organname, dataset='test')
    log_means(organ=args.organname, dataset='val')
    compute_ttest(results_dir='results_3090', organ=args.organname, dataset='val')
    compute_ttest(results_dir='results_3090', organ=args.organname, dataset='test')
