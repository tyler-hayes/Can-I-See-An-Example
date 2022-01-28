import numpy as np
import argparse
import os
import json
import math
import sys
import time
from copy import deepcopy

# imports from torch
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# imports from local scripts
from train_continual_active_learners.baseline_training import pre_train, get_pre_train_data_loaders
from train_continual_active_learners.evaluate import evaluate, get_predictions, evaluate_all_test_loaders
from train_continual_active_learners.triple_completion_model import CompletionModel
from train_continual_active_learners.triple_completion_utils import get_scores_and_truth, compute_split_sizes, \
    find_ix_a_in_b, find_set_diff, save_sample_distributions
from train_continual_active_learners.vg_qtype_dset import collate_fn_qtype
from train_continual_active_learners.vg_qtype_dset_loader_utils import build_loaders, get_generic_data_loader, \
    ReindexDataset, get_head_and_tail_test_loaders
from train_continual_active_learners.active_learning_utils import get_active_learning_data_loader, \
    choose_active_learning_samples
from train_continual_active_learners.bias_correction_model import BiasCorrectionModel

from utils import CMA, mkdir, randint, QType, TFLogger, bool_flag
from config import DATA_DIR, META_DATA_DIR


def get_args_parser(add_help=False):
    parser = argparse.ArgumentParser(description='Attribute & Predicate Prediction', add_help=add_help)

    # dataset options
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--shuffle_val', default=False, type=bool_flag)
    parser.add_argument('--loader_num_workers', default=0,
                        type=int)  # see https://github.com/pytorch/pytorch/issues/13246
    parser.add_argument('--vg_image_dir', default=os.path.join(DATA_DIR, ''))
    parser.add_argument('--train_h5', default=os.path.join(META_DATA_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(META_DATA_DIR, 'val.h5'))
    parser.add_argument('--test_h5', default=os.path.join(META_DATA_DIR, 'test.h5'))
    parser.add_argument('--vocab_json', default=os.path.join(META_DATA_DIR, 'vocab.json'))
    parser.add_argument('--load_images', default=False, type=bool_flag)  # dataloader returns images or not

    # box and question h5 files
    parser.add_argument('--train_box_feature_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_train.h5'))
    parser.add_argument('--val_box_feature_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_val.h5'))
    parser.add_argument('--test_box_feature_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_test.h5'))

    parser.add_argument('--train_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_coordinates_train.h5'))
    parser.add_argument('--val_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_coordinates_val.h5'))
    parser.add_argument('--test_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_coordinates_test.h5'))

    parser.add_argument('--train_question_h5', default=os.path.join(META_DATA_DIR, 'vg_qtype_dset_train.h5'))
    parser.add_argument('--val_question_h5', default=os.path.join(META_DATA_DIR, 'vg_qtype_dset_val.h5'))
    parser.add_argument('--test_question_h5', default=os.path.join(META_DATA_DIR, 'vg_qtype_dset_test.h5'))

    parser.add_argument('--train_triple_h5', default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_triples1_train.h5'))
    parser.add_argument('--val_triple_h5', default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_triples1_val.h5'))
    parser.add_argument('--test_triple_h5', default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_triples1_test.h5'))

    parser.add_argument('--train_attribute_h5',
                        default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_attributes1_train.h5'))
    parser.add_argument('--val_attribute_h5', default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_attributes1_val.h5'))
    parser.add_argument('--test_attribute_h5',
                        default=os.path.join(META_DATA_DIR, 'vg_scene_graphs_attributes1_test.h5'))

    parser.add_argument('--train_attribute_dict',
                        default=os.path.join(META_DATA_DIR, 'vg_qtype_attribute_dict_train.json'))
    parser.add_argument('--train_predicate_dict',
                        default=os.path.join(META_DATA_DIR, 'vg_qtype_predicate_dict_train.json'))
    parser.add_argument('--test_attribute_dict',
                        default=os.path.join(META_DATA_DIR, 'vg_qtype_attribute_dict_test.json'))
    parser.add_argument('--test_predicate_dict',
                        default=os.path.join(META_DATA_DIR, 'vg_qtype_predicate_dict_test.json'))

    # results and evaluation parameters
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--ckpt_epoch', type=int, default=1000)  # how often to save model ckpt
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--eval_only', type=bool_flag, default=False)
    parser.add_argument('--eval_head_and_tail', type=bool_flag, default=True)
    parser.add_argument('--test_tail', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num_attribute_categories', type=int, default=253)  # number of attribute classes
    parser.add_argument('--num_predicate_categories', type=int, default=46)

    # active learning parameters
    parser.add_argument('--active_learning_method', default=None, type=str,
                        choices=[None, 'random', 'confidence', 'entropy', 'margin', 'tail',
                                 'confidence_no_head', 'margin_no_head', 'entropy_no_head', 'tail_uniform_class',
                                 'tail_count_proba'])
    parser.add_argument('--num_active_learning_samples', type=int, default=600)
    parser.add_argument('--sampling_type', type=str, default='balanced_probabilistic',
                        choices=['probabilistic', 'rank', 'balanced', 'balanced_probabilistic', 'balanced_rank'])
    parser.add_argument('--num_active_learning_increments', type=int, default=10)

    # tail method active learning parameters
    parser.add_argument('--tail_seen_distribution', type=bool_flag, default=True)
    parser.add_argument('--tail_tail_probability', type=float, default=1.)
    parser.add_argument('--tail_head_probability', type=float, default=0.)

    # pre-train parameters
    parser.add_argument('--pre_training', default=False, type=bool_flag)
    parser.add_argument('--pre_train_iter_level', default=False, type=bool_flag)
    parser.add_argument('--num_head_samples_per_class', type=int, default=None)
    parser.add_argument('--class_permutation_seed', type=int, default=444)

    # cross-validation parameters
    parser.add_argument('--cross_validation_patience', type=int, default=10)
    parser.add_argument('--cross_validation_k', type=int, default=5)
    parser.add_argument('--cross_validate_selection', type=bool_flag, default=False)

    # hard negative mining parameters
    parser.add_argument('--hard_negative_top_k', type=int, default=3)
    parser.add_argument('--hard_negatives_type', type=str, default=None,
                        choices=[None, 'reservoir_hard', 'hard', 'random'])
    parser.add_argument('--num_hard_negatives', default=128,
                        type=int)  # this will be double (256) due to (positive, negative)
    parser.add_argument('--reset_hard_negative_reservoir_per_epoch', type=bool_flag, default=False)

    # network parameters
    parser.add_argument('--normalization', type=str, choices=['bn', 'ln'], default='bn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=256)
    parser.add_argument('--shared_dimension_size', type=int, default=128)
    parser.add_argument('--tied_target_weights', default=False, type=bool_flag)
    parser.add_argument('--normalize_embeddings', default=False, type=bool_flag)
    parser.add_argument('--embed_dist_metric', type=str, default='l2', choices=['l2', 'inner_prod', 'cosine'])
    parser.add_argument('--final_embedding_activation', type=str, default='sigmoid', choices=['sigmoid', 'mish'])
    parser.add_argument('--network_seed', type=int, default=0)

    # optimization parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_epoch', type=int, default=5)  # for computing validation loss during training (slow)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--use_standard_mini_batches', type=bool_flag, default=False)

    # batch sizes
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--al_batch_size', default=256, type=int)
    parser.add_argument('--pre_train_batch_size', default=800, type=int)
    parser.add_argument('--test_batch_size', default=512, type=int)

    # bias correction parameters
    parser.add_argument('--use_bias_correction', type=bool_flag, default=False)
    parser.add_argument('--bias_correction_class_lr', type=float, default=0.01)
    parser.add_argument('--bias_correction_class_epochs', type=int, default=500)
    parser.add_argument('--bias_correction_metric_lr', type=float, default=0.01)
    parser.add_argument('--bias_correction_metric_epochs', type=int, default=10)
    parser.add_argument('--bias_correction_metric_optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--bias_correction_class_optimizer', type=str, default='lbfgs', choices=['lbfgs', 'adam'])
    parser.add_argument('--bias_correction_two_stage', type=bool_flag, default=True)

    return parser


##############################################################################################
# MINI-BATCH AND OPTIMIZER HELPER FUNCTIONS

def find_non_pre_train_sample_ixs(model, questions, attr_labels, predicate_labels, device):
    # find indicies of new and old samples in current mini-batch for computing separate losses

    # get indices of attribute and predicate questions
    attr_q_ixs = torch.cat([torch.where(questions == QType.spas)[0], torch.where(questions == QType.spap)[0],
                            torch.where(questions == QType.spaa)[0]]).to(device)
    pred_q_ixs = torch.cat([torch.where(questions == QType.spos)[0], torch.where(questions == QType.spop)[0],
                            torch.where(questions == QType.spoo)[0]]).to(device)

    # find locations in attribute and predicate labels of visited classes
    old_attr_class_ixs = find_ix_a_in_b(attr_labels[attr_q_ixs],
                                        torch.from_numpy(model.visited_attribute_categories).to(device))
    old_attr_class_ixs = attr_q_ixs[old_attr_class_ixs]
    old_pred_class_ixs = find_ix_a_in_b(predicate_labels[pred_q_ixs],
                                        torch.from_numpy(model.visited_predicate_categories).to(device))
    old_pred_class_ixs = pred_q_ixs[old_pred_class_ixs]
    old_sample_ixs = torch.cat([old_attr_class_ixs, old_pred_class_ixs])
    full_ixs = torch.arange(len(questions)).to(device)
    new_class_ixs = find_set_diff(full_ixs, old_sample_ixs)
    return new_class_ixs, old_sample_ixs


def combine_batches(batch1, batch2):
    # combine two mini-batches into a single mini-batch

    mega_batch = {}
    offset = batch1['boxes'].shape[0]
    for (k1, v1), (k2, v2) in zip(batch1.items(), batch2.items()):
        if k1 not in ['subject_box_id', 'object_box_id', 'box_feature_ids_to_subject_box_ids']:
            if not isinstance(v1, list):
                if len(v1.shape) > 1:
                    mega_batch[k1] = torch.cat([v1, v2], dim=0)
                else:
                    mega_batch[k1] = torch.cat([v1, v2])
            else:
                mega_batch[k1] = v1 + v2
        else:
            mega_batch[k1] = torch.cat([v1, v2 + offset])
    return mega_batch


def get_optimizer_scheduler_criterion(args, model, loader):
    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.optimizer == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd, momentum=0.9)

    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader))
    else:
        raise NotImplementedError

    return optimizer, scheduler, criterion


##############################################################################################
# LOSS COMPUTATION HELPER FUNCTIONS

@torch.no_grad()
def compute_loss_new_and_old_samples(args, model, loader, criterion, device):
    model.eval()
    old_loss = CMA()
    new_loss = CMA()
    total_loss = CMA()
    for i, batch in enumerate(loader):
        scores, truth = get_scores_and_truth(args, model, batch, device)

        # determine which samples are old versus new
        predicate_labels = batch['predicates'].to(device)
        attr_labels = batch['attributes'].to(device)
        questions = batch['question_id'].to(device)
        new_ixs, old_ixs = find_non_pre_train_sample_ixs(model, questions, attr_labels, predicate_labels, device)
        if len(old_ixs) == 0:
            old_ixs = None
        if len(new_ixs) == 0:
            new_ixs = None

        loss = criterion(scores, truth)

        # update appropriate losses
        if new_ixs is not None:
            loss_new_samples = torch.mean(loss[new_ixs]).item()
            new_loss.update(loss_new_samples)
        if old_ixs is not None:
            loss_old_samples = torch.mean(loss[old_ixs]).item()
            old_loss.update(loss_old_samples)

        loss = torch.mean(loss).item()
        total_loss.update(loss)

    model.train()
    return total_loss.avg, old_loss.avg, new_loss.avg


@torch.no_grad()
def compute_rebalanced_mb_val_loss(args, model, prev_loader, new_loader, criterion, device):
    model.eval()
    val_loss = CMA()
    old_loss = CMA()
    new_loss = CMA()
    new_dataloader_iterator = iter(new_loader)

    for i, prev_batch in enumerate(prev_loader):  # assumes len(prev_loader) > len(new_loader)

        try:
            new_batch = next(new_dataloader_iterator)
        except StopIteration:
            new_dataloader_iterator = iter(new_loader)
            new_batch = next(new_dataloader_iterator)

        num_new_samples = len(new_batch['index'])
        num_old_samples = len(prev_batch['index'])
        new_ixs = torch.arange(num_new_samples)
        old_ixs = torch.arange(num_new_samples, num_new_samples + num_old_samples)

        batch = combine_batches(new_batch, prev_batch)
        scores, truth = get_scores_and_truth(args, model, batch, device)
        loss = criterion(scores, truth)
        loss_new_samples = torch.mean(loss[new_ixs]).item()
        if old_ixs is not None:
            loss_old_samples = torch.mean(loss[old_ixs]).item()
        else:
            loss_old_samples = np.inf
        loss = torch.mean(loss).item()

        val_loss.update(loss)
        old_loss.update(loss_old_samples)
        new_loss.update(loss_new_samples)
    model.train()
    return val_loss.avg, old_loss.avg, new_loss.avg


##############################################################################################
# STANDARD MINI-BATCH HELPER FUNCTIONS

def inner_training_loop_standard_mb(args, model, optimizer, criterion, batch, total_loss, new_loss, old_loss, device):
    # determine which samples are old versus new
    predicate_labels = batch['predicates'].to(device)
    attr_labels = batch['attributes'].to(device)
    questions = batch['question_id'].to(device)
    new_ixs, old_ixs = find_non_pre_train_sample_ixs(model, questions, attr_labels, predicate_labels, device)

    scores, truth = get_scores_and_truth(args, model, batch, device)
    loss = criterion(scores, truth)
    loss_new_samples = torch.mean(loss[new_ixs]).item()
    if old_ixs is not None:
        loss_old_samples = torch.mean(loss[old_ixs]).item()
    else:
        loss_old_samples = np.inf

    loss = torch.mean(loss)

    if math.isnan(loss.item()):
        print('\nEncountered NaN loss and exiting.')
        sys.exit()

    # update network
    optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
    loss.backward()
    optimizer.step()

    total_loss.update(loss.item())
    new_loss.update(loss_new_samples)
    old_loss.update(loss_old_samples)


def train_standard_mb(args, model, data_loader, test_loader, logger, device, verbose=True, unique_id=None, epochs=None,
                      val_epoch=5):
    if epochs is None:
        epoch = args.epochs
    else:
        epoch = epochs

    print('\nStarting training...')

    optimizer, scheduler, criterion = get_optimizer_scheduler_criterion(args, model, data_loader)

    # setup model
    model = model.to(device)
    model.train()

    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'
    msg_cluster = '\nEpoch (%d/%d) -- train_loss=%1.4f'

    for e in range(epoch):
        total_loss = CMA()
        old_loss = CMA()
        new_loss = CMA()

        for i, curr_batch in enumerate(data_loader):
            inner_training_loop_standard_mb(args, model, optimizer, criterion, curr_batch, total_loss, new_loss,
                                            old_loss, device)

            # if verbose:
            #     print(msg % (e + 1, epoch, i + 1, len(data_loader), total_loss.avg), end="")

        if verbose:
            print(msg_cluster % (e + 1, epoch, total_loss.avg))

        scheduler.step()

        # save losses for all increments
        if unique_id is not None:
            logger_name = 'loss_%s' % unique_id
        else:
            logger_name = 'loss'

        # compute val loss
        if e % val_epoch == 0:
            val_loss, val_old_loss, val_new_loss = compute_loss_new_and_old_samples(args, model, test_loader, criterion,
                                                                                    device)
            logger.add_scalar(logger_name + '/val', val_loss, e)
            logger.add_scalar(logger_name + '/val_old', val_old_loss, e)
            logger.add_scalar(logger_name + '/val_new', val_new_loss, e)

        logger.add_scalar(logger_name + '/train', total_loss.avg, e)
        logger.add_scalar(logger_name + '/train_old', old_loss.avg, e)
        logger.add_scalar(logger_name + '/train_new', new_loss.avg, e)

        if (e + 1) % args.ckpt_epoch == 0:
            # save ckpt every args.ckpt_epoch epochs
            d = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.avg,
            }
            if unique_id is None:
                name = 'triple_completion_model_epoch_%d_ckpt.pth' % e
            else:
                name = 'triple_completion_model_epoch_%d_id_%s_ckpt.pth' % (e, unique_id)

            torch.save(d, os.path.join(args.output_dir, name))

    # save final ckpt
    d = {
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.avg,
    }
    if unique_id is None:
        name = 'triple_completion_model_final_ckpt.pth'
    else:
        name = 'triple_completion_model_final_id_%s_ckpt.pth' % unique_id
    torch.save(d, os.path.join(args.output_dir, name))


def train_and_cross_validate_standard_mb(args, model, train_loader, val_loader, logger, device, verbose=True,
                                         patience=1, unique_id=None, use_best_val_loss=False):
    optimizer, scheduler, criterion = get_optimizer_scheduler_criterion(args, model, train_loader)

    # setup model
    model = model.to(device)
    model.train()

    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'
    msg_cluster = '\nEpoch (%d/%d) -- train_loss=%1.4f'

    best_val_loss = 10000
    best_epoch = args.epochs
    pc = 0  # patience counter
    early_stop = False

    for e in range(args.epochs):
        total_loss = CMA()
        old_loss = CMA()
        new_loss = CMA()

        for i, curr_batch in enumerate(train_loader):
            inner_training_loop_standard_mb(args, model, optimizer, criterion, curr_batch, total_loss, new_loss,
                                            old_loss, device)

            # if verbose:
            #     print(msg % (e + 1, args.epochs, i + 1, len(train_loader), total_loss.avg), end="")

        # compute validation loss and check patience
        val_loss, old_val_loss, new_val_loss = compute_loss_new_and_old_samples(args, model, val_loader, criterion,
                                                                                device)
        print(' -- val_loss=%1.4f' % val_loss)

        # save losses for all increments
        if unique_id is not None:
            logger_name = 'loss_%s' % str(unique_id)
        else:
            logger_name = 'loss'

        logger.add_scalar(logger_name + '/train', total_loss.avg, e)
        logger.add_scalar(logger_name + '/train_old', old_loss.avg, e)
        logger.add_scalar(logger_name + '/train_new', new_loss.avg, e)
        logger.add_scalar(logger_name + '/val', val_loss, e)
        if old_val_loss is not None:
            logger.add_scalar(logger_name + '/val_old', old_val_loss, e)
        if new_val_loss is not None:
            logger.add_scalar(logger_name + '/val_new', new_val_loss, e)

        if val_loss < best_val_loss and not math.isnan(val_loss):
            best_val_loss = val_loss
            best_epoch = (e + 1)
            pc = 0
        else:
            pc += 1
            if pc > patience and not use_best_val_loss:
                # patience counter exceeded patience, early stop
                print(' -- Early Stopping -- best_epoch=%d' % best_epoch)
                early_stop = True
                break
            print(' -- Patience=%d/%d' % (pc, patience))

        if verbose:
            print(msg_cluster % (e + 1, args.epochs, total_loss.avg))

        scheduler.step()

    # return best epoch
    if not early_stop and not use_best_val_loss:
        # if training did not early stop, return max epochs
        best_epoch = args.epochs
    return best_epoch


def cross_validate_standard_mb(args, model, al_dsets, logger, device, k=5, unique_id=None):
    # get ckpt before increment for cross-validation
    curr_model_ckpt = deepcopy(model.state_dict())

    # split data into k folds
    al_splits = []
    for dset in al_dsets:
        n_samples = len(dset)
        curr_sizes = compute_split_sizes(n_samples, k)
        al_splits.append(torch.utils.data.random_split(dset, curr_sizes, generator=torch.Generator().manual_seed(42)))

    # perform cross-validation and keep track of best epoch
    best_epochs = []
    for i in range(1):
        print('\nCross-validating iter %d/%d...' % (i + 1, k))

        # reset model
        model.load_state_dict(curr_model_ckpt)

        # save hold-out val set and loader
        val_set = torch.utils.data.ConcatDataset([s[i] for s in al_splits])
        val_set = ReindexDataset(val_set)
        val_loader = get_generic_data_loader(args, val_set, batch_size=args.test_batch_size, shuffle=True,
                                             drop_last=False, sampler=None)

        # make train set and loader
        train_set = []
        for j in range(k):
            if j != i:  # append non-val sets to pre-train data and previous active learning data
                train_set.extend([s[j] for s in al_splits])
        train_set = torch.utils.data.ConcatDataset(train_set)
        train_set = ReindexDataset(train_set)
        train_loader = get_generic_data_loader(args, train_set, batch_size=args.batch_size, shuffle=True,
                                               drop_last=False, sampler=None)

        e = train_and_cross_validate_standard_mb(args, model, train_loader, val_loader, logger, device,
                                                 patience=args.cross_validation_patience,
                                                 unique_id=unique_id + '_' + str(i),
                                                 use_best_val_loss=False)
        best_epochs.append(e)

    epochs = max(best_epochs)

    # reset model
    model.load_state_dict(curr_model_ckpt)
    return epochs


##############################################################################################
# HARD NEGATIVE MINING HELPER FUNCTIONS

@torch.no_grad()
def find_hard_negatives(args, model, batch, index, device, sort=False):
    scores, truth = get_scores_and_truth(args, model, batch, device)

    curr_k = min(args.hard_negative_top_k, len(scores))  # limited by size of mini-batch for k
    maxk = max((curr_k,))
    _, pred = scores.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(truth.view(1, -1).expand_as(pred))
    incorrect_ix = torch.where(torch.sum(correct, dim=0) == 0)[0]
    if len(incorrect_ix) == 0:
        return []
    else:
        if sort:
            scores_incorrect = scores[incorrect_ix]
            arr = torch.arange(len(scores_incorrect))
            scores_incorrect = scores_incorrect[arr, arr]
            sort_ix = torch.argsort(scores_incorrect)
            incorrect_ix = incorrect_ix[sort_ix]

        lhs = incorrect_ix.cpu()
        rhs = pred[:, lhs][0, :].cpu()

        lhs = index[lhs]
        rhs = index[rhs]
        ixs = []
        for l, r in zip(lhs, rhs):
            ixs.append((l.item(), r.item()))
        return ixs


def perform_hard_negative_mining(args, model, batch, prev_dataset, hard_negative_reservoir, device):
    index = batch['index']

    if 'reservoir' in args.hard_negatives_type:
        sort = False
    else:
        sort = True

    if args.hard_negatives_type == 'hard' or args.hard_negatives_type == 'reservoir_hard':
        hard_negative_ix = find_hard_negatives(args, model, batch, index, device, sort=sort)
    else:
        # random sample
        sample_ix = randint(len(index), args.num_hard_negatives * 2)
        sample_ix = index[np.array(sample_ix)]
        negatives_batch = []
        for select in sample_ix:
            negatives_batch.append(prev_dataset[select])
        return negatives_batch

    if 'reservoir' in args.hard_negatives_type:
        hard_negative_reservoir.extend(hard_negative_ix)
        size_hard_negatives = len(hard_negative_reservoir)

        # add some hard negatives to the batch
        selected_negatives = randint(size_hard_negatives, min(args.num_hard_negatives, size_hard_negatives))
        hard_negative_ix = [hard_negative_reservoir[neg_ix] for neg_ix in selected_negatives]
    else:
        # sample fixed number (hardest ones)
        hard_negative_ix = hard_negative_ix[:min(len(hard_negative_ix), args.num_hard_negatives)]

    negatives_batch = []
    if len(hard_negative_ix) > 0:
        for select in hard_negative_ix:
            # put both elements from tuple in batch
            negatives_batch.append(prev_dataset[select[0]])
            negatives_batch.append(prev_dataset[select[1]])
    return negatives_batch


##############################################################################################
# RE-BALANCED MINI-BATCH HELPER FUNCTIONS

def inner_training_loop_rebalanced_mb(args, model, optimizer, criterion, batch, prev_batch, prev_dataset,
                                      hard_negative_reservoir, total_loss, new_loss, old_loss, device):
    num_new_samples = len(batch['index'])
    new_ixs = torch.arange(num_new_samples)
    old_ixs = None
    if args.hard_negatives_type is not None:
        hard_negative_batch = perform_hard_negative_mining(args, model, prev_batch, prev_dataset,
                                                           hard_negative_reservoir, device)
        num_old_samples = len(hard_negative_batch)
        if num_old_samples > 0:
            hard_negative_batch = collate_fn_qtype(hard_negative_batch)
            batch = combine_batches(batch, hard_negative_batch)
            old_ixs = torch.arange(num_new_samples, num_new_samples + num_old_samples)

    scores, truth = get_scores_and_truth(args, model, batch, device)
    loss = criterion(scores, truth)
    loss_new_samples = torch.mean(loss[new_ixs]).item()
    if old_ixs is not None:
        loss_old_samples = torch.mean(loss[old_ixs]).item()
    else:
        loss_old_samples = np.inf

    loss = torch.mean(loss)

    if math.isnan(loss.item()):
        print('\nEncountered NaN loss and exiting.')
        sys.exit()

    # update network
    optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
    loss.backward()
    optimizer.step()

    total_loss.update(loss.item())
    new_loss.update(loss_new_samples)
    old_loss.update(loss_old_samples)


def train_rebalanced_mb(args, model, prev_data_loader, new_data_loader, test_loader, logger, device, verbose=True,
                        unique_id=None, epochs=None, val_epoch=5):
    if epochs is None:
        epoch = args.epochs
    else:
        epoch = epochs

    print('\nStarting training...')

    optimizer, scheduler, criterion = get_optimizer_scheduler_criterion(args, model, new_data_loader)

    prev_dataset = prev_data_loader.dataset

    # setup model
    model = model.to(device)
    model.train()

    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'
    msg_cluster = '\nEpoch (%d/%d) -- train_loss=%1.4f'

    hard_negative_reservoir = []

    prev_dataloader_iterator = iter(prev_data_loader)
    for e in range(epoch):
        total_loss = CMA()
        old_loss = CMA()
        new_loss = CMA()

        for i, curr_batch in enumerate(new_data_loader):

            try:
                prev_batch = next(prev_dataloader_iterator)
            except StopIteration:
                prev_dataloader_iterator = iter(prev_data_loader)
                prev_batch = next(prev_dataloader_iterator)

            inner_training_loop_rebalanced_mb(args, model, optimizer, criterion, curr_batch, prev_batch, prev_dataset,
                                              hard_negative_reservoir, total_loss, new_loss, old_loss, device)

            # if verbose:
            #     print(msg % (e + 1, epoch, i + 1, len(new_data_loader), total_loss.avg), end="")

        if verbose:
            print(msg_cluster % (e + 1, epoch, total_loss.avg))

        scheduler.step()

        print('\nHard Negative Reservoir Len: ', len(hard_negative_reservoir))

        if args.reset_hard_negative_reservoir_per_epoch:
            hard_negative_reservoir = []  # purge hard negatives at end of epoch

        # save losses for all increments
        if unique_id is not None:
            logger_name = 'loss_%s' % unique_id
        else:
            logger_name = 'loss'

        # compute val loss
        if e % val_epoch == 0:
            val_loss, val_old_loss, val_new_loss = compute_loss_new_and_old_samples(args, model, test_loader, criterion,
                                                                                    device)
            logger.add_scalar(logger_name + '/val', val_loss, e)
            logger.add_scalar(logger_name + '/val_old', val_old_loss, e)
            logger.add_scalar(logger_name + '/val_new', val_new_loss, e)

        logger.add_scalar(logger_name + '/train', total_loss.avg, e)
        logger.add_scalar(logger_name + '/train_old', old_loss.avg, e)
        logger.add_scalar(logger_name + '/train_new', new_loss.avg, e)

        if (e + 1) % args.ckpt_epoch == 0:
            # save ckpt every args.ckpt_epoch epochs
            d = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.avg,
            }
            if unique_id is None:
                name = 'triple_completion_model_epoch_%d_ckpt.pth' % e
            else:
                name = 'triple_completion_model_epoch_%d_id_%s_ckpt.pth' % (e, unique_id)

            torch.save(d, os.path.join(args.output_dir, name))

    # save final ckpt
    d = {
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.avg,
    }
    if unique_id is None:
        name = 'triple_completion_model_final_ckpt.pth'
    else:
        name = 'triple_completion_model_final_id_%s_ckpt.pth' % unique_id
    torch.save(d, os.path.join(args.output_dir, name))


def train_and_cross_validate_rebalanced_mb(args, model, prev_data_loader, al_train_loader, val_loader, logger, device,
                                           verbose=True, patience=1, unique_id=None, new_val_loader=None,
                                           use_best_val_loss=False):
    optimizer, scheduler, criterion = get_optimizer_scheduler_criterion(args, model, al_train_loader)

    # setup model
    model = model.to(device)
    model.train()

    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'
    msg_cluster = '\nEpoch (%d/%d) -- train_loss=%1.4f'

    best_val_loss = 10000
    best_epoch = args.epochs
    pc = 0  # patience counter
    early_stop = False
    hard_negative_reservoir = []

    prev_dataset = prev_data_loader.dataset
    prev_dataloader_iterator = iter(prev_data_loader)

    for e in range(args.epochs):
        total_loss = CMA()
        old_loss = CMA()
        new_loss = CMA()

        # iterate over new data loader
        for i, curr_batch in enumerate(al_train_loader):

            # iterate over previous data loader
            try:
                prev_batch = next(prev_dataloader_iterator)
            except StopIteration:
                prev_dataloader_iterator = iter(prev_data_loader)
                prev_batch = next(prev_dataloader_iterator)

            inner_training_loop_rebalanced_mb(args, model, optimizer, criterion, curr_batch, prev_batch, prev_dataset,
                                              hard_negative_reservoir, total_loss, new_loss, old_loss, device)

            # if verbose:
            #     print(msg % (e + 1, args.epochs, i + 1, len(al_train_loader), total_loss.avg), end="")

        # compute validation loss with new data in every batch and check patience
        val_loss, old_val_loss, new_val_loss = compute_rebalanced_mb_val_loss(args, model, val_loader, new_val_loader,
                                                                              criterion, device)
        print(' -- val_loss=%1.4f' % val_loss)

        # save losses for all increments
        if unique_id is not None:
            logger_name = 'loss_%s' % str(unique_id)
        else:
            logger_name = 'loss'

        logger.add_scalar(logger_name + '/train', total_loss.avg, e)
        logger.add_scalar(logger_name + '/train_old', old_loss.avg, e)
        logger.add_scalar(logger_name + '/train_new', new_loss.avg, e)
        logger.add_scalar(logger_name + '/val', val_loss, e)
        if old_val_loss is not None:
            logger.add_scalar(logger_name + '/val_old', old_val_loss, e)
        if new_val_loss is not None:
            logger.add_scalar(logger_name + '/val_new', new_val_loss, e)

        if val_loss < best_val_loss and not math.isnan(val_loss):
            best_val_loss = val_loss
            best_epoch = (e + 1)
            pc = 0
        else:
            pc += 1
            if pc > patience and not use_best_val_loss:
                # patience counter exceeded patience, early stop
                print(' -- Early Stopping -- best_epoch=%d' % best_epoch)
                early_stop = True
                break
            print(' -- Patience=%d/%d' % (pc, patience))

        if verbose:
            print(msg_cluster % (e + 1, args.epochs, total_loss.avg))

        scheduler.step()

        if args.reset_hard_negative_reservoir_per_epoch:
            hard_negative_reservoir = []  # purge hard negatives at end of epoch

    # return best epoch
    if not early_stop and not use_best_val_loss:
        # if training did not early stop, return max epochs
        best_epoch = args.epochs
    return best_epoch


def cross_validate_rebalanced_mb(args, model, al_dsets, logger, device, k=5, unique_id=None):
    # get ckpt before increment for cross-validation
    curr_model_ckpt = deepcopy(model.state_dict())

    # split data into k folds
    al_splits = []
    for dset in al_dsets:
        n_samples = len(dset)
        curr_sizes = compute_split_sizes(n_samples, k)
        al_splits.append(torch.utils.data.random_split(dset, curr_sizes, generator=torch.Generator().manual_seed(42)))

    # ratio of old samples to new samples
    ratio_old_to_new = args.pre_train_batch_size / args.al_batch_size

    # perform cross-validation and keep track of best epoch
    best_epochs = []
    for i in range(1):
        print('\nCross-validating iter %d/%d...' % (i + 1, k))

        # reset model
        model.load_state_dict(curr_model_ckpt)

        # save hold-out val sets and loaders
        al_val_set = al_splits[-1][i]
        al_val_set = ReindexDataset(al_val_set)
        al_batch_size = min(len(al_val_set), args.al_batch_size)
        al_val_loader = get_generic_data_loader(args, al_val_set, batch_size=al_batch_size, shuffle=True,
                                                drop_last=False)
        old_val_set = torch.utils.data.ConcatDataset([s[i] for s in al_splits[:-1]])
        old_val_set = ReindexDataset(old_val_set)
        old_batch_size = int(al_batch_size * ratio_old_to_new)
        old_val_loader = get_generic_data_loader(args, old_val_set, batch_size=old_batch_size,
                                                 shuffle=True, drop_last=False)
        # val_set = torch.utils.data.ConcatDataset([al_val_set, old_val_set])
        # val_set = ReindexDataset(val_set)
        # val_loader = get_generic_data_loader(args, val_set, batch_size=args.test_batch_size, shuffle=True,
        #                                      drop_last=False)

        # make train sets and loaders
        al_train_set = []
        old_train_set = []
        for j in range(k):
            if j != i:  # append non-val sets to pre-train data and previous active learning data
                al_train_set.append(al_splits[-1][j])
                old_train_set.extend([s[j] for s in al_splits[:-1]])
        al_train_set = torch.utils.data.ConcatDataset(al_train_set)
        al_train_set = ReindexDataset(al_train_set)
        al_batch_size = min(len(al_val_set), args.al_batch_size)
        al_train_loader = get_generic_data_loader(args, al_train_set, batch_size=al_batch_size, shuffle=True,
                                                  drop_last=True)

        old_train_set = torch.utils.data.ConcatDataset(old_train_set)
        old_train_set = ReindexDataset(old_train_set)
        old_batch_size = int(al_batch_size * ratio_old_to_new)
        old_train_loader = get_generic_data_loader(args, old_train_set,
                                                   batch_size=old_batch_size, shuffle=True,
                                                   drop_last=True)

        e = train_and_cross_validate_rebalanced_mb(args, model, old_train_loader, al_train_loader, old_val_loader,
                                                   logger, device, patience=args.cross_validation_patience,
                                                   unique_id=unique_id + '_' + str(i), new_val_loader=al_val_loader,
                                                   use_best_val_loss=False)

        best_epochs.append(e)

    epochs = max(best_epochs)

    # reset model
    model.load_state_dict(curr_model_ckpt)
    return epochs


def make_rebalanced_dataloaders(args, previous_active_learn_datasets, ratio_old_to_new):
    al_dataset = previous_active_learn_datasets[-1]  # only most recent AL data is new
    al_dataset = ReindexDataset(al_dataset)
    al_batch_size = min(len(al_dataset), args.al_batch_size)
    curr_train_loader = get_active_learning_data_loader(args, al_dataset, batch_size=al_batch_size)

    # combine (pre-train data, all previous AL data)
    pre_train_batch_size = int(al_batch_size * ratio_old_to_new)
    pre_train_dset = torch.utils.data.ConcatDataset(previous_active_learn_datasets[:-1])
    pre_train_dset = ReindexDataset(pre_train_dset)
    prev_train_loader = get_active_learning_data_loader(args, pre_train_dset, batch_size=pre_train_batch_size)
    return curr_train_loader, prev_train_loader


##############################################################################################
# BIAS CORRECTION HELPER FUNCTIONS

def full_bias_correction(args, model, train_loader, inc, device):
    print('\nPerforming bias correction...')

    # stage 1: correct metric space
    bias_correct_metric_space(args, model, train_loader, device=device, num_iters=args.bias_correction_metric_epochs,
                              lr=args.bias_correction_metric_lr)

    if args.bias_correction_two_stage:
        # stage 2: correct predicate and attribute class distances for SPOP and SPAA
        preds, truth = get_predictions(args, model, train_loader, device)

        bc_model = BiasCorrectionModel(num_predicates=args.num_predicate_categories,
                                       num_attributes=args.num_attribute_categories, lr=args.bias_correction_class_lr,
                                       num_iters=args.bias_correction_class_epochs, use_alpha_attribute=True,
                                       use_alpha_predicate=True, optimizer=args.bias_correction_class_optimizer)
        for (key_pred, val_pred), (key_true, val_true) in zip(preds.items(), truth.items()):
            if key_pred == 0 or key_pred == 1 or key_pred == 4 or key_pred == 3:  # skip box-based and spap questions
                continue
            print('\n', key_pred)
            if key_pred == 2:
                bc_model.train_bias_correction_parameters(val_pred, val_true, type='predicates')
            else:
                bc_model.train_bias_correction_parameters(val_pred, val_true, type='attributes')

        # save model and bias correction parameters
        d = {'model_state_dict': model.state_dict(),
             'bias_correction_model_state_dict': bc_model.state_dict()}
        name = 'triple_completion_model_with_bias_correction_final_inc_%d_ckpt.pth' % inc
        torch.save(d, os.path.join(args.output_dir, name))
    else:
        d = {'model_state_dict': model.state_dict()}
        name = 'triple_completion_model_with_bias_correction_final_inc_%d_ckpt.pth' % inc
        torch.save(d, os.path.join(args.output_dir, name))
        bc_model = None

    return bc_model


def bias_correct_metric_space(args, model, data_loader, num_iters=10, lr=0.01, device='cuda'):
    print('\nTraining bias correction on metric space...')
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'

    if args.bias_correction_metric_optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader))
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    for e in range(num_iters):
        total_loss = CMA()
        for i, batch in enumerate(data_loader):
            scores, truth = get_scores_and_truth(args, model, batch, device)
            loss = criterion(scores, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            print(msg % (e + 1, num_iters, i + 1, len(data_loader), total_loss.avg), end="")
        scheduler.step()


##############################################################################################
# MAIN FUNCTIONS

def incremental_active_learning(args, model, main_logger, cross_val_logger, device):
    # get pre-train data loader and unlabeled pool
    vocab, train_loader, unlabeled_train_loader, pre_train_labeled_loader, val_loader, test_loader, class_ix_dict = build_loaders(
        args, shuffle=False)
    unlabel_dataset = unlabeled_train_loader.dataset

    # optionally get head and tail test loaders
    if args.eval_head_and_tail:
        head_test_loader, tail_test_loader = get_head_and_tail_test_loaders(args, test_loader.dataset)
    else:
        head_test_loader = None
        tail_test_loader = None

    # save pre-train sample distributions
    save_sample_distributions(args, pre_train_labeled_loader, model, 'pre_train_sample_distributions.json')

    # put pre-train classes in visited lists on model
    model.visited_predicate_categories = class_ix_dict['predicates']
    model.visited_attribute_categories = class_ix_dict['attributes']

    # ratio of pre-train samples to new samples
    ratio_old_to_new = args.pre_train_batch_size / args.al_batch_size

    # perform incremental active learning in this loop
    previous_active_learn_datasets = [pre_train_labeled_loader.dataset]
    for inc in range(args.num_active_learning_increments):

        print('\nChoosing active learning samples for increment %d/%d...' % (
            inc + 1, args.num_active_learning_increments))

        # get unlabeled loader
        unlabel_dataset = ReindexDataset(unlabel_dataset)
        current_unlabeled_loader = get_generic_data_loader(args, unlabel_dataset, batch_size=args.test_batch_size,
                                                           shuffle=False)

        # compute active learning model scores over unlabeled dataset
        # orig_active_ixs are chosen indices w.r.t. to original full training set for easier saving and re-loading
        active_dataset, unlabel_dataset, active_ixs, unlabel_ixs, orig_active_ixs = choose_active_learning_samples(args,
                                                                                                                   model,
                                                                                                                   args.active_learning_method,
                                                                                                                   current_unlabeled_loader,
                                                                                                                   device=device)
        # save active ixs out to json files
        save_file = os.path.join(args.output_dir, 'active_learn_increment_%d_chosen_ixs.json' % inc)
        with open(save_file, 'w', encoding="utf8") as f:
            json.dump(orig_active_ixs.tolist(), f)

        # save sample distribution to file
        curr_active_learning_loader = get_generic_data_loader(args, active_dataset, batch_size=args.test_batch_size,
                                                              shuffle=False)
        save_sample_distributions(args, curr_active_learning_loader, model,
                                  'active_learn_increment_%d_sample_distributions.json' % inc)
        previous_active_learn_datasets.append(active_dataset)

        if not args.use_standard_mini_batches:
            print('\nTraining with re-balanced mini-batches...')

            if args.cross_validate_selection:
                epochs = cross_validate_rebalanced_mb(args, model, previous_active_learn_datasets, cross_val_logger,
                                                      device, k=args.cross_validation_k, unique_id=str(inc))
            else:
                epochs = args.epochs

            # train model on combination of new and old data
            curr_train_loader, prev_train_loader = make_rebalanced_dataloaders(args, previous_active_learn_datasets,
                                                                               ratio_old_to_new)
            train_rebalanced_mb(args, model, prev_train_loader, curr_train_loader, test_loader, main_logger, device,
                                verbose=True, unique_id=str(inc), epochs=epochs, val_epoch=args.val_epoch)

        else:
            print('\nTraining with standard mini-batches...')

            if args.cross_validate_selection:
                epochs = cross_validate_standard_mb(args, model, previous_active_learn_datasets, cross_val_logger,
                                                    device,
                                                    k=args.cross_validation_k, unique_id=str(inc))
            else:
                epochs = args.epochs

            # train model on combination of new and old data
            train_dset = torch.utils.data.ConcatDataset(previous_active_learn_datasets)
            train_dset = ReindexDataset(train_dset)
            train_loader = get_active_learning_data_loader(args, train_dset, batch_size=args.batch_size,
                                                           sampler=None)

            train_standard_mb(args, model, train_loader, test_loader, main_logger, device, verbose=True,
                              unique_id=str(inc), epochs=epochs, val_epoch=args.val_epoch)

        # optionally train bias correction network
        if args.use_bias_correction:

            # copy model weights
            curr_model_ckpt = deepcopy(model.state_dict())

            # compute performance before bias correction
            evaluate_all_test_loaders(args, inc, epochs, model, test_loader, head_test_loader,
                                      tail_test_loader, None, device, prefix='before_bias_correction_')

            print('\nPerforming bias correction...')
            bias_train_dset = torch.utils.data.ConcatDataset(previous_active_learn_datasets)
            bias_train_dset = ReindexDataset(bias_train_dset)
            bias_train_loader = get_active_learning_data_loader(args, bias_train_dset, batch_size=args.batch_size)

            # perform bias correction and evaluate model
            bias_correction_model = full_bias_correction(args, model, bias_train_loader, inc, device)
            curr_results_dict = evaluate_all_test_loaders(args, inc, epochs, model, test_loader, head_test_loader,
                                                          tail_test_loader, bias_correction_model, device)

            # set model back to initial state and unfreeze parameters
            model.load_state_dict(curr_model_ckpt)
            model.assign_network_plasticity_all_params(require_grad=True)
        else:
            curr_results_dict = evaluate_all_test_loaders(args, inc, epochs, model, test_loader, head_test_loader,
                                                          tail_test_loader, None, device)

    return curr_results_dict


def main(args):
    print(args)
    device = args.device

    # setup directories for saving results
    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)

    tf_logging_dir = os.path.join(args.output_dir, 'tf_logging')
    if not os.path.exists(tf_logging_dir):
        mkdir(tf_logging_dir)
    cross_val_dir = os.path.join(tf_logging_dir, 'cross_val')
    if not os.path.exists(cross_val_dir):
        mkdir(cross_val_dir)
    train_dir = os.path.join(tf_logging_dir, 'main')
    if not os.path.exists(train_dir):
        mkdir(train_dir)

    main_logger = TFLogger(train_dir)
    cross_val_logger = TFLogger(cross_val_dir)

    # setup network layer shapes
    if args.num_layers == 1:
        bbox_encoder_shape = [1024, args.shared_dimension_size]
        prediction_network_shape = [3 * args.shared_dimension_size, args.shared_dimension_size]
    else:
        hidden = []
        for i in range(args.num_layers - 1):
            hidden.append(args.num_neurons)

        bbox_encoder_shape = [1024] + hidden + [args.shared_dimension_size]
        prediction_network_shape = [3 * args.shared_dimension_size] + hidden + [args.shared_dimension_size]

    # get model
    torch.manual_seed(args.network_seed)
    model = CompletionModel(bbox_encoder_shape=bbox_encoder_shape, prediction_network_shape=prediction_network_shape,
                            num_predicates=args.num_predicate_categories, num_attributes=args.num_attribute_categories,
                            normalize_embeddings=args.normalize_embeddings,
                            tied_target_weights=args.tied_target_weights, dropout_prob=args.dropout,
                            final_activation=args.final_embedding_activation, normalization=args.normalization)

    # load pre-trained ckpt
    if args.ckpt_file is not None:
        print('\nLoading ckpt from: %s' % args.ckpt_file)
        ckpt_file = torch.load(args.ckpt_file)
        state_dict = ckpt_file['model_state_dict']
        model.load_state_dict(state_dict)

    # train and evaluate model
    if not args.eval_only:
        start_time = time.time()
        if args.pre_training:
            # get data
            vocab, train_loader, val_loader, test_loader, class_ix = get_pre_train_data_loaders(args, drop_last=True)
            pre_train(args, model, train_loader, test_loader, main_logger, device, verbose=True,
                      iteration_level=args.pre_train_iter_level)

            # evaluate model with learned classifiers
            results_dict = evaluate(args, model, test_loader, device)
        else:
            # active learning
            results_dict = incremental_active_learning(args, model, main_logger, cross_val_logger, device)
        print('\nTrain+Eval Time: ', time.time() - start_time)
    else:
        if args.test_tail:
            vocab, train_loader, val_loader, test_loader, class_ix = get_pre_train_data_loaders(args, drop_last=True)
            head_test_loader, tail_test_loader = get_head_and_tail_test_loaders(args, test_loader.dataset)

            # evaluate model with learned classifiers
            results_dict = evaluate_all_test_loaders(args, -1, None, model, test_loader, head_test_loader,
                                                     tail_test_loader, None, device)
        else:
            vocab, train_loader, val_loader, test_loader, class_ix = get_pre_train_data_loaders(args, drop_last=True)
            # evaluate model with learned classifiers
            results_dict = evaluate(args, model, test_loader, device)

    if not args.eval_only:
        # save results to json
        json.dump(results_dict, open(os.path.join(args.output_dir, 'final_results.json'), 'w'))


if __name__ == '__main__':
    start = time.time()
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    print('Total time: ', time.time() - start)
