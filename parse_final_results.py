import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pylab
from collections import defaultdict

from utils import QType

key = {0: 'SPOS', 1: 'SPAS', 2: 'SPOP', 3: 'SPAP', 4: 'SPOO', 5: 'SPAA'}
ylims = {0: [0.3, 0.4], 1: [0.25, 0.35], 2: [0, 0.1], 3: [0.85, 1.0], 4: [0.3, 0.35], 5: [0, 0.05]}


def save_plot_to_file(x_vals, mega_results_dict, mu_baseline, std_baseline, mu_offline, std_offline, metric,
                      include_std=True,
                      include_full=True, include_pt=True, save_plot=False, plots_dir=None, plot_name=None,
                      exclude_spop_spaa_upper_bound=False, fontsize=18):
    qtypes = [1, 5, 0, 2, 4]
    for qtype in qtypes:
        if qtype == 3:
            continue
        include_upper = include_full
        if exclude_spop_spaa_upper_bound:
            if qtype == 2 or qtype == 5:
                include_upper = False

        # plot active learning results
        markers = ['p', 's', 'D', 'o', '>', '^', '<']
        colors = ['#999999', '#377eb8', '#4daf4a',
                  '#f781bf', '#984ea3',
                  '#ff7f00', '#e41a1c', '#dede00']
        baseline_colors = ['b', 'r', 'k']

        linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(0, len(x_vals))
        for i, (k, v) in enumerate(mega_results_dict.items()):
            v_mu = v[0][qtype, :]
            v_stdev = v[1][qtype, :]
            ax.plot(x, v_mu, marker=markers[i], color=colors[i], linestyle=linestyles[i], linewidth=2, markersize=7,
                    label=k)
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=colors[i], alpha=0.4)

        # baseline results
        if include_pt:
            v_mu = np.ones_like(x) * mu_baseline[qtype]
            v_stdev = np.ones_like(x) * std_baseline[qtype]
            ax.plot(x, v_mu, color=baseline_colors[1], linestyle='dashed', linewidth=2, markersize=7, label='Pre-Train')
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=baseline_colors[1], alpha=0.5)

        if include_upper:
            v_mu = np.ones_like(x) * mu_offline[qtype]
            v_stdev = np.ones_like(x) * std_offline[qtype]
            ax.plot(x, v_mu, color=baseline_colors[-1], linestyle='dashed', linewidth=2, markersize=7, label='Full')
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=baseline_colors[-1], alpha=0.5)

        ax.set_xlabel('Increment', fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(metric, fontweight='bold', fontsize=fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(x_vals, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        handles, labels = ax.get_legend_handles_labels()
        ax.grid()
        plt.tight_layout()
        if save_plot:
            curr_name = plot_name % (metric + '_' + key[qtype])
            print('Saving %s...' % curr_name)
            fig.savefig(os.path.join(plots_dir, curr_name + '.png'), bbox_inches="tight", format='png')
            fig = pylab.figure()
            figlegend = pylab.figure(figsize=(17, 1))
            ax = fig.add_subplot(111)
            figlegend.legend(handles, labels, fontsize=fontsize, loc='center', fancybox=True,
                             shadow=True, ncol=7)
            figlegend.savefig(os.path.join(plots_dir, curr_name + '_legend.png'), bbox_inches="tight", format='png')


def make_plot(x_vals, mega_results_dict, mu_baseline, std_baseline, mu_offline, std_offline, metric, include_std=True,
              include_full=True, include_pt=True, show_plots=False):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 9))  # (7, 4.8)
    l = [ax1, ax2, ax3, ax4, ax5, ax6]
    qtypes = [1, 3, 5, 0, 2, 4]
    summary_results = defaultdict(list)
    for qtype, ax in zip(qtypes, l):
        if qtype == 3:
            continue

        # plot active learning results
        markers = ['p', 's', 'D', 'o', '>', '^', '<']
        colors = ['#999999', '#377eb8', '#4daf4a',
                  '#f781bf', '#984ea3',
                  '#ff7f00', '#e41a1c', '#dede00']
        baseline_colors = ['b', 'r', 'k']

        linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

        x = np.arange(0, len(x_vals))
        for i, (k, v) in enumerate(mega_results_dict.items()):
            v_mu = v[0][qtype, :]
            v_stdev = v[1][qtype, :]
            ax.plot(x, v_mu, marker=markers[i], color=colors[i], linestyle=linestyles[i], linewidth=2, markersize=7,
                    label=k)
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=colors[i], alpha=0.5)

            summary_stat = compute_summary_statistic(mu_offline[qtype], v_mu)
            summary_results[k].append(summary_stat)

        # baseline results
        if include_pt:
            v_mu = np.ones_like(x) * mu_baseline[qtype]
            v_stdev = np.ones_like(x) * std_baseline[qtype]
            ax.plot(x, v_mu, color=baseline_colors[1], linestyle='dashed', linewidth=2, markersize=7, label='Pre-Train')
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=baseline_colors[1], alpha=0.5)

        if include_full:
            v_mu = np.ones_like(x) * mu_offline[qtype]
            # print('\nOffline ', key[qtype], v_mu)
            v_stdev = np.ones_like(x) * std_offline[qtype]
            ax.plot(x, v_mu, color=baseline_colors[-1], linestyle='dashed', linewidth=2, markersize=7, label='Full')
            if include_std:
                ax.fill_between(x, v_mu - v_stdev, v_mu + v_stdev, color=baseline_colors[-1], alpha=0.5)

        ax.set_xlabel('Increment', fontweight='bold', fontsize=16)
        ax.set_ylabel(metric, fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(x_vals, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        handles, labels = ax.get_legend_handles_labels()
        ax.set_title('QType: ' + key[qtype], fontsize=18)
        ax.grid()

    if include_full:
        ncol = len(mega_results_dict) + 2
    else:
        ncol = len(mega_results_dict) + 1
    fig.legend(handles, labels, fontsize=16, ncol=ncol, loc='lower center', fancybox=True,
               shadow=True, bbox_to_anchor=[0.55, -0.001])
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    if show_plots:
        plt.show()

    print('\nMetric: ', metric)
    for k, v in summary_results.items():
        print('\n', k + ' & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f' % (v[0], v[1], v[2], v[3], v[4]))


def compute_summary_statistic(offline, active_learner_performance):
    return np.mean(1 - (offline - active_learner_performance))


def get_results(job_id_dict, save_dir, return_epoch=False, macro=False):
    results_dict_roc_auc = {}
    results_dict_map = {}
    results_dict_epochs = {}
    for k, v in job_id_dict.items():
        mu_roc_auc, mu_map, epochs = make_results_one_run(v, save_dir, return_epoch=return_epoch, macro=macro)
        mu_roc_auc = np.transpose(np.array(mu_roc_auc))
        mu_map = np.transpose(np.array(mu_map))
        results_dict_roc_auc[k] = mu_roc_auc
        results_dict_map[k] = mu_map
        results_dict_epochs[k] = epochs
    return results_dict_roc_auc, results_dict_map, results_dict_epochs


def make_results_one_run(specific_dirs, save_dir, return_epoch=False, macro=False):
    roc_auc = []
    map = []
    epochs = []
    for specific_dir in specific_dirs:
        roc_auc_, map_, e_ = get_one_set_of_results(specific_dir, save_dir, return_epoch=return_epoch, macro=macro)
        roc_auc.append(roc_auc_)
        map.append(map_)
        epochs.append(e_)
    return roc_auc, map, epochs


def get_one_set_of_results(specific_dir, save_dir, return_epoch=False, macro=False):
    auroc = []
    mean_ap = []
    if os.path.exists(save_dir % specific_dir):
        with open(save_dir % specific_dir, 'r') as f:
            final_results = json.load(f)
        if return_epoch:
            e = final_results['epochs']
        for qtype in QType:
            if macro and qtype == QType.spop:
                ix_roc = 4
                ix_map = 5
            elif macro and qtype == QType.spaa:
                ix_roc = 2
                ix_map = 3
            else:
                ix_roc = 0
                ix_map = 1
            qtype = str(int(qtype))
            auroc.append(final_results[qtype][ix_roc])
            mean_ap.append(final_results[qtype][ix_map])
    else:
        # if file doesn't exist, append a nan
        for qtype in QType:
            auroc.append(np.nan)
            mean_ap.append(np.nan)
        if return_epoch:
            e = np.nan

    if return_epoch:
        return auroc, mean_ap, e
    else:
        return auroc, mean_ap, None


def get_distribution_results(job_id_dict, save_dir):
    results_dict = {}
    for k, v in job_id_dict.items():
        result = make_all_distributions(v, save_dir)
        results_dict[k] = result
    return results_dict


def make_all_distributions(specific_dirs, save_dir):
    result_list = []
    for specific_dir in specific_dirs:
        result = get_distributions(specific_dir, save_dir)
        result_list.append(result)
    return result_list


def get_distributions(specific_dir, save_dir):
    with open(save_dir % specific_dir, 'r') as f:
        final_results = json.load(f)
    return final_results


def build_methods_dict(name, methods, method_names, num_increments, file_name):
    d = {}
    for m, m_pretty in zip(methods, method_names):
        name_list = []
        for inc in range(num_increments):
            f = os.path.join(name % m, file_name % inc)
            name_list.append(f)
        d[m_pretty] = name_list
    return d


def parse_results(mega_auroc_results_list, mega_map_results_list, method_names, total_runs, num_questions,
                  num_check_increments):
    mega_auroc_dict = {}
    mega_map_dict = {}
    for method in method_names:
        auroc_curr = np.zeros((total_runs, num_questions, num_check_increments))
        map_curr = np.zeros((total_runs, num_questions, num_check_increments))

        # get result for each run and put into arrays
        for it in range(len(mega_auroc_results_list)):
            auroc_curr[it, :, :] = mega_auroc_results_list[it][method]
            map_curr[it, :, :] = mega_map_results_list[it][method]

        # put mean and stdev arrays into mega dictionaries
        mega_auroc_dict[method] = (np.mean(auroc_curr, axis=0), np.std(auroc_curr, axis=0) / np.sqrt(total_runs))
        mega_map_dict[method] = (np.mean(map_curr, axis=0), np.std(map_curr, axis=0) / np.sqrt(total_runs))
    return mega_auroc_dict, mega_map_dict


def get_results_dict(seeds, permutations, al_first_name, al_second_name, methods, method_names, num_check_increments,
                     save_dir, file_name, macro):
    mega_auroc_results_list = []
    mega_map_results_list = []
    mega_epoch_list = []
    for seed in seeds:
        for perm in permutations:
            al_name = al_first_name + al_second_name % (seed, perm)

            methods_dict = build_methods_dict(al_name, methods, method_names, num_check_increments, file_name)

            roc_auc, mean_ap, epochs = get_results(methods_dict, save_dir, return_epoch=False, macro=macro)
            mega_auroc_results_list.append(roc_auc)
            mega_map_results_list.append(mean_ap)
            mega_epoch_list.append(epochs)
    return mega_auroc_results_list, mega_map_results_list, mega_epoch_list


def get_all_results(save_dir, plots_dir, seeds, num_questions, permutations, total_runs, num_increments,
                    offline_baseline_name, pre_train_baseline_name, baseline_file_name, methods, method_names,
                    plot_name, al_first_name, al_second_name, file_name_al, save_plots=False, show_plots=False,
                    include_pt=True, include_full=True, include_std=True, exclude_spop_spaa_upper_bound=False):
    ########################################################################################################
    # offline baseline results

    auroc_base_full = np.zeros((len(seeds), num_questions))
    map_base_full = np.zeros((len(seeds), num_questions))
    it = 0
    for seed in seeds:
        baseline_name = offline_baseline_name % seed
        baseline_name = os.path.join(baseline_name, baseline_file_name % -1)
        auroc_base, map_base, _ = get_one_set_of_results(baseline_name, save_dir, macro=False)
        auroc_base_full[it, :] = auroc_base
        map_base_full[it, :] = map_base
        it += 1
    mu_auroc_base = np.mean(auroc_base_full, axis=0)
    std_auroc_base = np.std(auroc_base_full, axis=0) / np.sqrt(total_runs)
    mu_map_base = np.mean(map_base_full, axis=0)
    std_map_base = np.std(map_base_full, axis=0) / np.sqrt(total_runs)

    ########################################################################################################
    # pre-train baseline results

    auroc_pt_full = np.zeros((total_runs, num_questions))
    map_pt_full = np.zeros((total_runs, num_questions))
    it = 0
    for seed in seeds:
        for perm in permutations:
            baseline_name = pre_train_baseline_name % (seed, perm)
            baseline_name = os.path.join(baseline_name, baseline_file_name % -1)
            auroc_pt, map_pt, _ = get_one_set_of_results(baseline_name, save_dir, macro=False)
            auroc_pt_full[it, :] = auroc_pt
            map_pt_full[it, :] = map_pt
            it += 1
    mu_auroc_pt = np.mean(auroc_pt_full, axis=0)
    std_auroc_pt = np.std(auroc_pt_full, axis=0) / np.sqrt(total_runs)
    mu_map_pt = np.mean(map_pt_full, axis=0)
    std_map_pt = np.std(map_pt_full, axis=0) / np.sqrt(total_runs)

    ########################################################################################################
    # active learning results

    mega_auroc_results_list, mega_map_results_list, mega_epoch_list = get_results_dict(seeds, permutations,
                                                                                       al_first_name,
                                                                                       al_second_name,
                                                                                       methods, method_names,
                                                                                       num_increments,
                                                                                       save_dir,
                                                                                       file_name_al, False)

    mega_auroc_dict, mega_map_dict = parse_results(mega_auroc_results_list, mega_map_results_list,
                                                   method_names, total_runs, num_questions, num_increments)

    ########################################################################################################
    # plotting

    x_vals = [str(s) for s in range(1, num_increments + 1)]
    make_plot(x_vals, mega_auroc_dict, mu_auroc_pt, std_auroc_pt, mu_auroc_base, std_auroc_base, 'AUROC',
              include_std=include_std, include_full=include_full, include_pt=include_pt, show_plots=show_plots)
    make_plot(x_vals, mega_map_dict, mu_map_pt, std_map_pt, mu_map_base, std_map_base, 'mAP',
              include_std=include_std,
              include_full=include_full, include_pt=include_pt, show_plots=show_plots)

    if save_plots:
        save_plot_to_file(x_vals, mega_auroc_dict, mu_auroc_pt, std_auroc_pt, mu_auroc_base, std_auroc_base, 'AUROC',
                          include_std=include_std, include_full=include_full, include_pt=include_pt,
                          save_plot=save_plots, plots_dir=plots_dir, plot_name=plot_name)
        save_plot_to_file(x_vals, mega_map_dict, mu_map_pt, std_map_pt, mu_map_base, std_map_base, 'mAP',
                          include_std=include_std, include_full=include_full, include_pt=include_pt,
                          save_plot=save_plots, plots_dir=plots_dir, plot_name=plot_name,
                          exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)

    return mega_auroc_dict, mega_map_dict


def plot_rebalanced_mb_with_bias_correction(test_data, num_samples, head_samples):
    methods = ['random', 'confidence', 'entropy', 'margin', 'tail_rank_equal_proba']
    method_names = ['Random', 'Confidence', 'Entropy', 'Margin', 'Tail (Ours)']
    plot_name = 'final_' + test_data + '_results_rebalanced_mb_%s'

    al_first_name = 'final_active_learning_head_pre_train_bal_qtype_method_%s_old_new_bias_correction_' + str(
        num_samples) + '_num_head_samples_' + str(head_samples) + '_network_seed_'
    al_second_name = '%d_permutation_seed_%d'

    file_name = test_data + '_results_increment_%d.json'
    return methods, method_names, plot_name, al_first_name, al_second_name, file_name


def plot_rebalanced_mb_without_bias_correction(test_data, num_samples, head_samples):
    methods = ['random', 'confidence', 'entropy', 'margin', 'tail_rank_equal_proba']
    method_names = ['Random', 'Confidence', 'Entropy', 'Margin', 'Tail (Ours)']
    plot_name = 'final_' + test_data + '_results_rebalanced_mb_without_bias_correction_%s'

    al_first_name = 'final_active_learning_head_pre_train_bal_qtype_method_%s_old_new_bias_correction_' + str(
        num_samples) + '_num_head_samples_' + str(head_samples) + '_network_seed_'
    al_second_name = '%d_permutation_seed_%d'

    file_name = 'before_bias_correction_' + test_data + '_results_increment_%d.json'
    return methods, method_names, plot_name, al_first_name, al_second_name, file_name


def plot_al_tail_only(test_data, num_samples, head_samples):
    methods = ['confidence_no_head', 'entropy_no_head', 'margin_no_head', 'tail_rank_equal_proba']
    method_names = ['Confidence', 'Entropy', 'Margin', 'Tail (Ours)']
    plot_name = 'final_' + test_data + '_results_rebalanced_mb_tail_only_%s'

    al_first_name = 'final_active_learning_head_pre_train_bal_qtype_method_%s_old_new_bias_correction_' + str(
        num_samples) + '_num_head_samples_' + str(head_samples) + '_network_seed_'
    al_second_name = '%d_permutation_seed_%d'

    file_name = test_data + '_results_increment_%d.json'
    return methods, method_names, plot_name, al_first_name, al_second_name, file_name


def plot_tail_method_comparisons(test_data, num_samples, head_samples):
    methods = ['tail_rank', 'tail_rank_equal_proba']
    method_names = ['Tail (Count Probabilities)', 'Tail (Full Setup)']
    plot_name = 'final_' + test_data + '_results_rebalanced_mb_tail_comparison_%s'

    al_first_name = 'final_active_learning_head_pre_train_bal_qtype_method_%s_old_new_bias_correction_' + str(
        num_samples) + '_num_head_samples_' + str(head_samples) + '_network_seed_'
    al_second_name = '%d_permutation_seed_%d'

    file_name = test_data + '_results_increment_%d.json'
    return methods, method_names, plot_name, al_first_name, al_second_name, file_name


def plot_standard_mb(test_data, num_samples, head_samples):
    methods = ['random', 'confidence', 'entropy', 'margin', 'tail_rank_equal_proba']
    method_names = ['Random', 'Confidence', 'Entropy', 'Margin', 'Tail (Ours)']
    plot_name = 'final_' + test_data + '_results_standard_mb_%s'

    al_first_name = 'final_active_learning_head_pre_train_bal_qtype_method_%s_standard_mini_batch_' + str(
        num_samples) + '_num_head_samples_' + str(head_samples) + '_network_seed_'
    al_second_name = '%d_permutation_seed_%d'

    file_name = test_data + '_results_increment_%d.json'
    return methods, method_names, plot_name, al_first_name, al_second_name, file_name


def main():
    ### CHANGE THIS PATH FOR SAVING PLOTS
    save_dir = '/media/tyler/Data/codes/Long-Tail-Active-Learning/results/%s'

    plots_dir = save_dir % 'plots'
    num_samples = 600  # number of active learning samples
    head_samples = 2500  # number of samples from each head class during pre-training
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # network seeds tested
    permutations = [444]  # permutation seeds tested
    num_questions = 6  # number of question types
    num_increments = 10  # number of training increments
    total_runs = len(seeds) * len(permutations)

    include_full = True  # include full offline upper bound in plot
    include_std = True  # include standard error in plot
    include_pt = True  # include pre-train baseline in plot

    # pre-train and offline baseline file names
    pre_train_baseline_name = 'triple_completion_baseline_sgd_num_head_iter_samples_' + str(
        head_samples) + '_network_seed_%d_permutation_seed_%d'
    offline_baseline_name = 'triple_completion_full_dataset_baseline_sgd_network_seed_%d'

    # get/plot results on full test set and tail test set for each experiment
    for test_data in ['full', 'tail']:
        print('\nTest Data: ', test_data)

        # exclude upper bounds from tail plots for clarity
        if test_data == 'tail':
            exclude_spop_spaa_upper_bound = True
        else:
            exclude_spop_spaa_upper_bound = False

        # pre-train and offline file name
        baseline_file_name = test_data + '_results_increment_%d.json'

        # plot re-balanced mini-batch results (with bias correction)
        print('\nRe-balanced with bias correction')
        methods, method_names, plot_name, al_first_name, al_second_name, file_name_al = plot_rebalanced_mb_with_bias_correction(
            test_data, num_samples, head_samples)
        auroc_dict_rebal_bc, map_dict_rebal_bc = get_all_results(save_dir, plots_dir, seeds, num_questions,
                                                                 permutations, total_runs, num_increments,
                                                                 offline_baseline_name, pre_train_baseline_name,
                                                                 baseline_file_name, methods, method_names,
                                                                 plot_name, al_first_name, al_second_name, file_name_al,
                                                                 save_plots=True,
                                                                 include_pt=include_pt, include_full=include_full,
                                                                 include_std=include_std,
                                                                 exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)

        # plot re-balanced mini-batch results (without bias correction)
        print('\nRe-balanced without bias correction')
        methods, method_names, plot_name, al_first_name, al_second_name, file_name_al = plot_rebalanced_mb_without_bias_correction(
            test_data, num_samples, head_samples)
        auroc_dict_rebal_no_bc, map_dict_rebal_no_bc = get_all_results(save_dir, plots_dir, seeds, num_questions,
                                                                       permutations, total_runs, num_increments,
                                                                       offline_baseline_name, pre_train_baseline_name,
                                                                       baseline_file_name, methods, method_names,
                                                                       plot_name, al_first_name, al_second_name,
                                                                       file_name_al,
                                                                       include_pt=include_pt, include_full=include_full,
                                                                       include_std=include_std,
                                                                       exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)

        # plot tail only results for all methods
        print('\nTail Only Results')
        methods, method_names, plot_name, al_first_name, al_second_name, file_name_al = plot_al_tail_only(
            test_data, num_samples, head_samples)
        auroc_dict_tail_data_only, map_dict_tail_data_only = get_all_results(save_dir, plots_dir, seeds, num_questions,
                                                                             permutations, total_runs, num_increments,
                                                                             offline_baseline_name,
                                                                             pre_train_baseline_name,
                                                                             baseline_file_name, methods, method_names,
                                                                             plot_name, al_first_name, al_second_name,
                                                                             file_name_al,
                                                                             include_pt=include_pt,
                                                                             include_full=include_full,
                                                                             include_std=include_std,
                                                                             exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)

        # plot tail method comparisons
        print('\nTail Comparisons')
        methods, method_names, plot_name, al_first_name, al_second_name, file_name_al = plot_tail_method_comparisons(
            test_data, num_samples, head_samples)
        auroc_dict_tail_comparisons, map_dict_tail_comparisons = get_all_results(save_dir, plots_dir, seeds,
                                                                                 num_questions,
                                                                                 permutations, total_runs,
                                                                                 num_increments,
                                                                                 offline_baseline_name,
                                                                                 pre_train_baseline_name,
                                                                                 baseline_file_name, methods,
                                                                                 method_names,
                                                                                 plot_name, al_first_name,
                                                                                 al_second_name, file_name_al,
                                                                                 include_pt=include_pt,
                                                                                 include_full=include_full,
                                                                                 include_std=include_std,
                                                                                 exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)

        # plot standard mini-batch results
        print('\nStandard Mini-Batches')
        methods, method_names, plot_name, al_first_name, al_second_name, file_name_al = plot_standard_mb(
            test_data, num_samples, head_samples)
        auroc_dict_standard, map_dict_standard = get_all_results(save_dir, plots_dir, seeds, num_questions,
                                                                 permutations, total_runs, num_increments,
                                                                 offline_baseline_name, pre_train_baseline_name,
                                                                 baseline_file_name, methods, method_names,
                                                                 plot_name, al_first_name, al_second_name, file_name_al,
                                                                 include_pt=include_pt, include_full=include_full,
                                                                 include_std=include_std,
                                                                 exclude_spop_spaa_upper_bound=exclude_spop_spaa_upper_bound)


if __name__ == '__main__':
    main()
