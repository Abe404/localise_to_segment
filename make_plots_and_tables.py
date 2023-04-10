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
import string
from scipy.stats.stats import pearsonr
import nibabel as nib

from skimage.transform import resize
from skimage import img_as_float
from skimage.io import imsave
import shutil
import matplotlib as mpl
from collections import defaultdict
from compute_results import get_converged_runs

mpl.rcParams["font.family"] = "sans-serif"

import matplotlib.pyplot as plt
import seaborn as sns
import texfig

ORGANS = ['spleen', 'pancreas', 'prostate', 'liver', 'heart']

from csv_utils import load_csv

def multi_plot_dice(exp_output_dir, plot_output_path):
    # 5 rows and 3 columns with width of 20 inches and height of 30 inches
    fig, axs = plt.subplots(5, 3, figsize=(20, 30))
    plt.rcParams['font.size'] = 24
    clrs = sns.color_palette("husl", 2)
    train_color = clrs[0]
    val_color = clrs[1]

    organs_dices = {}
    variants = ['full', 'low_res', 'cropped_by_org']
    for i, organ in enumerate(ORGANS):
        axs[i, 0].set_ylabel(organ, fontsize=26,  labelpad=10)

        for j, variant in enumerate(variants):
            log_dir = f'{exp_output_dir}/{organ}_{variant}'
            runs_dir = os.path.join(log_dir, 'runs')

            axs[i, j].grid(True)
            axs[i, j].set_yticks(np.arange(0, 1.025, 0.2))
            axs[i, j].set_xticks(np.arange(0, 300, 25))
            axs[i, j].set_ylim(0, 1.0)
            axs[i, j].set_xlim(0, 190)
            axs[i, j].tick_params(labelsize=30)

            for r_idx, run_dir in enumerate(os.listdir(runs_dir)):

                train_csv = os.path.join(runs_dir, run_dir, 'train_metrics.csv')
                dices = load_csv(train_csv, ['dice'], [float])[0]
                dices = np.array(dices)
                dices[np.isnan(dices)] = 0
                epochs = range(len(dices))

                if i == len(ORGANS) - 1 and j == 2 and r_idx == 0:
                    dice_label = f'Training'
                else:
                    dice_label = None

                p = axs[i, j].plot(epochs, dices, label=dice_label, color=train_color)

                #prev_color = p[0].get_color()
                val_csv = os.path.join(runs_dir, run_dir, 'val_metrics.csv')
                dices = load_csv(val_csv, ['dice'], [float])[0]
                dices = np.array(dices)
                dices[np.isnan(dices)] = 0
                epochs = range(len(dices))
                p = None
                if i == len(ORGANS) - 1 and j == 2 and r_idx == 0:
                    dice_label = f'Validation'
                else:
                    dice_label = None

                axs[i, j].plot(epochs, dices, color=val_color, label=dice_label, linestyle='--')
                if dice_label:
                    axs[i, j].legend(loc='lower right')

    fig.supylabel('dice', weight='bold')
    fig.supxlabel('epochs', y=0.02)
    
    axs[0, 0].set_title('baseline', pad=20)
    axs[0, 1].set_title('low res', pad=20)
    axs[0, 2].set_title('organ', pad=20)

    plt.grid(True)
    plt.tight_layout()
    print('saving figure to ', plot_output_path)
    # plt.savefig(fig_path)
    texfig.savefig(plot_output_path)

def box_plot(baseline_csv, twostage_csv, gt_loc_csv, output_plot_fname):
    plt.figure(figsize=(4,6))
    baseline_dices = load_csv(baseline_csv, ['dice'], [float])[0]
    twostage_dices = load_csv(twostage_csv, ['dice'], [float])[0]
    gt_loc_dices = load_csv(gt_loc_csv, ['dice'], [float])[0]

    plt.scatter([1] * len(baseline_dices), baseline_dices, marker='x')
    plt.scatter([2] * len(twostage_dices), twostage_dices, marker='x')
    plt.scatter([3] * len(gt_loc_dices), gt_loc_dices, marker='x')
    plt.xticks([1, 2, 3], ['baseline', 'two stage', 'ground truth localised'], rotation=45)
    plt.xlim((0.5, 3.5))
    plt.tight_layout()
    texfig.savefig(output_plot_fname)


def get_dices_over_epochs(log_dir):
    """ return two lists of lists.
        first list of lists is a list of train dices (for each run and then for each epoch).
        second list of lists is a list of val dices (for each run and then for each epoch). """

    runs_dir = os.path.join(log_dir, 'runs')
    val_runs = []
    train_runs = []

    for run_dir in os.listdir(runs_dir):
        train_csv = os.path.join(runs_dir, run_dir, 'train_metrics.csv')
        _train_seconds, train_dices = load_csv(train_csv,
                                               ['seconds', 'dice'],
                                               [float, float])
        train_dices = np.array(train_dices)
        train_dices[np.isnan(train_dices)] = 0
        train_epochs = range(len(train_dices))
        val_csv = os.path.join(runs_dir, run_dir, 'val_metrics.csv')
        _, val_dices = load_csv(val_csv,
                                ['seconds', 'dice'],
                                [float, float])
        val_dices = np.array(val_dices)
        val_dices[np.isnan(val_dices)] = 0
        val_epochs = range(len(val_dices))
        val_runs.append(val_dices)
        train_runs.append(train_dices)
    return val_runs, train_runs


def convergence_count(ax, exp_output_dir, clrs):
    # how many runs until convergence for each method and each dataset.
    from compute_results import get_converged_runs 
    variants = ['full', 'low_res', 'cropped_by_org']

    # dict with list for each variant
    # the variant list includes how many times each organ failed for that method variant
    organ_runs_failed = defaultdict(list)

    for i, organ in enumerate(ORGANS):
        for j, variant in enumerate(variants):
            log_dir = f'{exp_output_dir}/{organ}_{variant}'
            converged, all_runs = get_converged_runs(log_dir)
            all_runs = [int(a) for a in all_runs]
            converged = [int(a) for a in converged]
            all_failed = [a for a in all_runs if a not in converged]
            first_10_converged = sorted(converged)[:10]
            failed_before_10_converged = [a for a in all_failed if a < max(first_10_converged)]
            assert len(first_10_converged) == 10
            converged = first_10_converged
            failed = failed_before_10_converged
            total_runs = max(first_10_converged) # how many runs until 10 converged
            percent = 100 - round(((len(failed) / total_runs) * 100))
            organ_runs_failed[organ].append(percent)
            print('For the ', organ, variant, len(failed), 'runs failed before 10 converged, '
                  'giving a success percent of', percent)

    print('organ runs failed low res mean', np.mean([organ_runs_failed[a][1] for a in organ_runs_failed]))
    print('organ runs failed full mean', np.mean([organ_runs_failed[a][0] for a in organ_runs_failed]))

    # making a grouped bar chart.
    x = np.arange(len(variants))  # the label locations
    width = 0.15   # the width of the bars
    multiplier = 0
    ax.grid(axis='y')
    bars = []
    for i, (attribute, measurement) in enumerate(organ_runs_failed.items()):
        offset = (width * multiplier) 
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=clrs[i])
        ax.bar_label(rects, padding=2, fontsize=8)
        bars.append(rects)
        multiplier += 1
    ax.set_ylabel('Run convergence (%)', weight='bold')
    ax.set_xticks(x + (width*2), ['baseline', 'low res', 'organ'])
    return bars, ORGANS 

    
def scatter_plot_size_vs_converge(ax, clrs):
    spleen = sum([82, 100, 100]) / 3
    pancreas = sum([75, 82, 100]) / 3
    prostate = sum([82, 75, 100]) / 3
    liver = sum([100, 100, 100]) / 3
    heart = sum([75, 69, 100]) / 3

    spleen_count = 41
    pancreas_count = 282
    prostate_count = 32
    liver_count = 131
    heart_count = 20
    x = [spleen_count, pancreas_count, prostate_count, liver_count, heart_count]
    y = [spleen, pancreas, prostate, liver, heart]
    # n = ['spleen', 'pancreas', 'prostate', 'liver', 'heart']
    ax.grid()
    ax.set_xticks((0, 50, 100, 150, 200, 250, 300))
    ax.set_xlim((-10, 310))
    ax.scatter(x, y, color=clrs)
    ax.set_xlabel('dataset size')


def scatter_plot_class_balance_vs_converge(ax, clrs):
    spleen = sum([82, 100, 100]) / 3
    pancreas = sum([75, 82, 100]) / 3
    prostate = sum([82, 75, 100]) / 3
    liver = sum([100, 100, 100]) / 3
    heart = sum([75, 69, 100]) / 3

    spleen_percent = 0.4
    pancreas_percent = 0.2
    heart_percent = 0.4
    prostate_percent = 2.7
    liver_percent = 2.3
     
    x = [spleen_percent, pancreas_percent, prostate_percent, liver_percent, heart_percent]
    y = [spleen, pancreas, prostate, liver, heart]

    n = ['spleen', 'pancreas', 'prostate', 'liver', 'heart']
    
    ax.grid()
    ax.scatter(x, y, color=clrs)
    ax.set_xlabel('class balance (foreground %)')
    ax.set_xlim(-0.1, 3.1)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3.0])


def combined_convergence_plot(exp_dir):
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style('white'):
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained', gridspec_kw={'width_ratios': [1.6, 1, 1]})
        plt.tight_layout()
        bars, labels = convergence_count(axs[0], 'exp_output_3090', clrs)
        scatter_plot_size_vs_converge(axs[1], clrs)
        scatter_plot_class_balance_vs_converge(axs[2], clrs)
        axs[1].legend(bars, labels, loc='best')
        ylim = (67, 103)
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)
        axs[2].set_ylim(ylim)
        plt.tight_layout()
        axs = axs.flat
        axs[0].text(0.012, 0.95, '(a)',  transform=axs[0].transAxes, size=10, weight='bold')
        axs[1].text(0.035, 0.95, '(b)',  transform=axs[1].transAxes, size=10, weight='bold')
        axs[2].text(0.035, 0.95, '(c)',  transform=axs[2].transAxes, size=10, weight='bold')
        texfig.savefig('latex_plots/convergence_stats')


def training_time_comparison(exp_output_dir):
    durations = {}
    for variant in ['full', 'low_res', 'cropped_by_org']:
        durations[variant] = {}
        for organ in ORGANS:
            durations[variant][organ] = []
            log_dir = f'{exp_output_dir}/{organ}_{variant}'
            converged, all_runs = get_converged_runs(log_dir)
            runs_dir = os.path.join(log_dir, 'runs')
            for i, run_dir in enumerate(converged):
                train_csv = os.path.join(runs_dir, run_dir, 'train_metrics.csv')
                seconds, dices = load_csv(train_csv,
                                          ['seconds', 'dice'],
                                          [float, float])
                minutes = [s/60 for s in seconds]
                duration = minutes[-1]
                #print(organ, minutes)
                assert duration == max(minutes)
                durations[variant][organ].append(duration)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(durations['full'].values(), vert=False)
    ax.set_yticklabels(ORGANS)
    ax.grid()
    table_str = r"""
        \begin{table}[]
        \begin{tabular}{llll}
        \hline
                 & baseline & low res & cropped by organ \\ \hline
        """
    for organ in ORGANS:
        table_str += f"{organ} & {round(np.mean(durations['full'][organ]), 1)} "
        table_str += f"& {round(np.mean(durations['low_res'][organ]), 1)} & {round(np.mean(durations['cropped_by_org'][organ]), 1)} \\\\"
        table_str += "\n"
    table_str += r"""\hline'
            \end{tabular}
            \end{table}
        """
    print(table_str, file=open('results/train_time_table.tex', 'w+'))



def make_results_table(dataset='val'):

    results_dir = 'results_3090'
    table_str = r"""
        \begin{table}[H]
        \caption{\label{table:""" + dataset + """dice}"""
    table_str += f"Average dice on the {dataset} set for the baseline network"
    table_str += "compared to the two stage approach with both predicted and ground truth localisation.}"
    table_str += r"""
        \begin{center}
        \begin{tabular}{@{}llll@{}}
        \toprule
                 & baseline & two_stage & organ (groung truth localized) \\ \midrule
        """

    for organ in ORGANS:
        means_csv = open(f'results_3090/{organ}_{dataset}_means.csv').readlines()[1:]
        baseline_mean = means_csv[0].split(',')[1]
        baseline_std = means_csv[0].split(',')[2]

        two_stage_mean = means_csv[1].split(',')[1]
        two_stage_std = means_csv[1].split(',')[2]

        gt_loc_mean = means_csv[2].split(',')[1]
        gt_loc_std = means_csv[2].split(',')[2]
        table_str += (f"{organ} & {baseline_mean} $\pm$ {baseline_std} "
                      f"& {two_stage_mean} $\pm$ {two_stage_std} "
                      f"& {gt_loc_mean} $\pm$ {gt_loc_std} \\\\")
        table_str += "\n"
    table_str += r""" \bottomrule
            \hline
            \end{tabular}
            \end{center}
            \end{table}
        """

    print(table_str, file=open(f'results_3090/{dataset}_dice_table.tex', 'w+'))


def plot_class_balance_vs_localisation_benefit():
    names = ['spleen', 'pancreas', 'prostate', 'liver', 'heart']
    percents = [0.43780517578125,
                0.21805809823654426, 
                2.7093768054989127,
                2.2849260059115855,
                0.4020698308839718]

    heart_two_stage_minus_baseline_dice_mean = 0.046599999999999975
    heart_two_stage_gt_loc_minus_baseline_dice_mean = 0.1744
    spleen_two_stage_minus_baseline_dice_mean = 0.20699999999999996
    spleen_two_stage_gt_loc_minus_baseline_dice_mean = 0.3822
    pancreas_two_stage_minus_baseline_dice_mean = 0.21530000000000005
    pancreas_two_stage_gt_loc_minus_baseline_dice_mean = 0.30310000000000004
    prostate_two_stage_minus_baseline_dice_mean = -0.0766
    prostate_two_stage_gt_loc_minus_baseline_dice_mean = 0.03909999999999991
    liver_two_stage_minus_baseline_dice_mean = 0.005700000000000038
    liver_two_stage_gt_loc_minus_baseline_dice_mean = 0.11170000000000002

    diffs_two_stage_baseline = [
        spleen_two_stage_minus_baseline_dice_mean,
        pancreas_two_stage_minus_baseline_dice_mean,
        prostate_two_stage_minus_baseline_dice_mean,
        liver_two_stage_minus_baseline_dice_mean,
        heart_two_stage_minus_baseline_dice_mean
    ]

    diffs_gt_loc_baseline = [
        spleen_two_stage_gt_loc_minus_baseline_dice_mean,
        pancreas_two_stage_gt_loc_minus_baseline_dice_mean,
        prostate_two_stage_gt_loc_minus_baseline_dice_mean,
        liver_two_stage_gt_loc_minus_baseline_dice_mean,
        heart_two_stage_gt_loc_minus_baseline_dice_mean
    ]
    plt.figure()
    clrs = sns.color_palette("husl", 5)
    for i, label in enumerate(names):
        plt.scatter(diffs_two_stage_baseline[i], percents[i], label=label + ' two stage', color=clrs[i])
    max_x = 4.01
    max_y = max(diffs_gt_loc_baseline)
    for i, label in enumerate(names):
        plt.scatter(diffs_gt_loc_baseline[i], percents[i], label=label + ' gt loc', color=clrs[i], marker='x')
    plt.xlabel('mean localised dice - mean baseline dice')
    plt.ylabel('foreground %')
    plt.legend()
    plt.grid()
    texfig.savefig('latex_plots/foreground_percent_vs_gt_loc_benefit')

if __name__ == '__main__':
    plot_class_balance_vs_localisation_benefit()
    training_time_comparison('exp_output_3090')
    make_results_table(dataset='val')
    make_results_table(dataset='test')
    combined_convergence_plot('exp_output_3090')
    multi_plot_dice('exp_output_3090', f'latex_plots/training')
