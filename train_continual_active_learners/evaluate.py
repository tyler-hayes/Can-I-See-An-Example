import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from train_continual_active_learners.triple_completion_utils import compute_dist_scores, create_questions_for_batch
from utils import QType


def compute_metrics(truth, predicted, metric_type='macro', verbose=False):
    mean_ap = average_precision_score(truth, predicted, average=metric_type)
    roc_score = roc_auc_score(truth, predicted, average=metric_type)

    if verbose:
        print('\nROC AUC Score: %0.4f' % roc_score)
        print('\nmAP Score: %0.4f' % mean_ap)
    return roc_score, mean_ap


def compute_box_metrics(y_test, y_score):
    average_precision = []
    roc_auc = []

    for i in range(len(y_test)):
        y_test_i = y_test[i][0].flatten()
        y_score_i = y_score[i][0].flatten()
        average_precision.append(average_precision_score(y_test_i, y_score_i, average='samples'))
        roc_auc.append(roc_auc_score(y_test_i, y_score_i, average='samples'))
    average_precision = np.mean(np.array(average_precision))
    roc_auc = np.mean(np.array(roc_auc))
    return roc_auc, average_precision


def compute_scores(args, ques_triples, ques_target, predicted_features, target_features, curr_batch_box_features,
                   predicate_embeddings, attr_embeddings, predicate_labels, attr_labels,
                   box_feature_ids_to_subject_box_ids, results_dict_predicted_scores, results_dict_truth, index,
                   include_index=False, bias_correction_model=None, device=None):
    if args.use_bias_correction and bias_correction_model is not None:
        bias_correction_model.to(device)
        bias_correction_model.eval()

    for q_type in QType:
        q_ix = np.where(ques_target[:, 0] == q_type)[0]
        if len(q_ix) == 0:
            # no questions of this type, continue
            continue
        curr_target = ques_target[q_ix]
        curr_triples = ques_triples[q_ix]
        curr_pred_feats = predicted_features[q_ix]
        curr_target_feats = target_features[q_ix]
        curr_qid = index[q_ix]

        if q_type == QType.spos or q_type == QType.spas:

            # get dists for each feature based on box features in same scene graph. also return idx of true box feature
            for j in range(len(curr_target)):
                subject_box_id = curr_target[:, 1][j]
                curr_pred_feat = curr_pred_feats[j].unsqueeze(0)
                idxs = torch.where(box_feature_ids_to_subject_box_ids == subject_box_id)[0]
                boxes_in_graph = curr_batch_box_features[idxs]
                dists = compute_dist_scores(args.embed_dist_metric, boxes_in_graph, curr_pred_feat)
                truth_location_in_dists = torch.where(idxs == subject_box_id)[0]
                if include_index:
                    predicted = [curr_qid[j], dists.cpu().numpy()]
                else:
                    predicted = [dists.cpu().numpy()]
                results_dict_predicted_scores[q_type].append(predicted)
                results_dict_truth[q_type].append([
                    torch.nn.functional.one_hot(truth_location_in_dists,
                                                num_classes=dists.shape[-1]).cpu().numpy()])

        elif q_type == QType.spoo:

            # get dists for each feature based on box features in same scene graph. also return idx of true box feature
            for j in range(len(curr_triples)):
                subject_box_id = curr_triples[:, 1][j]
                answer_object_box_id = curr_target[:, 1][j]
                curr_pred_feat = curr_pred_feats[j].unsqueeze(0)
                idxs = torch.where(box_feature_ids_to_subject_box_ids == subject_box_id)[0]
                boxes_in_graph = curr_batch_box_features[idxs]
                dists = compute_dist_scores(args.embed_dist_metric, boxes_in_graph, curr_pred_feat)
                truth_location_in_dists = torch.where(idxs == answer_object_box_id)[0]
                if include_index:
                    predicted = [curr_qid[j], dists.cpu().numpy()]
                else:
                    predicted = [dists.cpu().numpy()]
                results_dict_predicted_scores[q_type].append(predicted)
                results_dict_truth[q_type].append([
                    torch.nn.functional.one_hot(truth_location_in_dists,
                                                num_classes=dists.shape[-1]).cpu().numpy()])

        elif q_type == QType.spop or q_type == QType.spap:
            dists = compute_dist_scores(args.embed_dist_metric, predicate_embeddings, curr_pred_feats)

            # optionally add bias correction to dists
            if args.use_bias_correction and bias_correction_model is not None and q_type == QType.spop:
                # only correct for spop
                dists = bias_correction_model.correct_biases(dists, type='predicates')

            labels_truth = predicate_labels[curr_target[:, -1]]

            for j, (curr_pred_feat, curr_true_feat, dist, curr_labels) in enumerate(
                    zip(curr_pred_feats, curr_target_feats, dists, labels_truth)):
                num_predicate_categories = args.num_predicate_categories
                if include_index:
                    predicted = [curr_qid[j], dist.cpu().numpy()]
                else:
                    predicted = [dist.cpu().numpy()]
                results_dict_predicted_scores[q_type].append(predicted)
                results_dict_truth[q_type].append([
                    torch.nn.functional.one_hot(curr_labels,
                                                num_classes=num_predicate_categories).cpu().numpy()])

        elif q_type == QType.spaa:
            dists = compute_dist_scores(args.embed_dist_metric, attr_embeddings, curr_pred_feats)

            # optionally add bias correction to dists
            if args.use_bias_correction and bias_correction_model is not None:
                dists = bias_correction_model.correct_biases(dists, type='attributes')

            labels_truth = attr_labels[curr_target[:, -1]]

            for j, (curr_pred_feat, curr_true_feat, dist, curr_labels) in enumerate(
                    zip(curr_pred_feats, curr_target_feats, dists, labels_truth)):
                num_attribute_categories = args.num_attribute_categories
                if include_index:
                    predicted = [curr_qid[j], dist.cpu().numpy()]
                else:
                    predicted = [dist.cpu().numpy()]
                results_dict_predicted_scores[q_type].append(predicted)
                results_dict_truth[q_type].append([
                    torch.nn.functional.one_hot(curr_labels,
                                                num_classes=num_attribute_categories).cpu().numpy()])
        else:
            raise NotImplementedError


def get_model_scores(args, model, data_loader, device, include_index=False, bias_correction_model=None):
    # setup model
    model = model.to(device)
    model.eval()

    # keys are q_types and value is list of distance_scores/one_hot_encoded_truth
    results_dict_predicted_scores = defaultdict(list)
    results_dict_truth = defaultdict(list)

    # compute embeddings for all one-hot combinations of predicates and attributes
    pred_batch = torch.arange(args.num_predicate_categories).to(device)
    attr_batch = torch.arange(args.num_attribute_categories).to(device)
    predicate_embeddings = model.compute_predicate_embeddings(pred_batch, target=True)
    attr_embeddings = model.compute_attribute_embeddings(attr_batch, target=True)

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        indices = batch['index']
        question_id = batch['question_id']
        box_features = batch['box_features']
        subject_box_id = batch['subject_box_id']
        object_box_id = batch['object_box_id']
        predicate_labels = batch['predicates']
        attr_labels = batch['attributes']
        box_feature_ids_to_subject_box_ids = batch['box_feature_ids_to_subject_box_ids']

        # setup data and feed through model
        bbox_batch = box_features.to(device)  # num_boxes x bbox_feat_size
        ques_triples, ques_target = create_questions_for_batch(args, question_id, subject_box_id, object_box_id)
        predicate_labels = predicate_labels.to(device)
        attr_labels = attr_labels.to(device)

        # compute network predictions to questions and target predictions
        predicted_features = model(bbox_batch, predicate_labels, attr_labels, ques_triples, device)
        target_features = model.get_target_features(bbox_batch, predicate_labels, attr_labels, ques_target, device)

        # get features for all boxes in the batch
        curr_batch_box_features = model.compute_bbox_embeddings(bbox_batch.to(device), target=True)

        # compute model scores
        compute_scores(args, ques_triples, ques_target, predicted_features, target_features,
                       curr_batch_box_features, predicate_embeddings, attr_embeddings, predicate_labels, attr_labels,
                       box_feature_ids_to_subject_box_ids, results_dict_predicted_scores, results_dict_truth, indices,
                       include_index=include_index, device=device, bias_correction_model=bias_correction_model)

    return results_dict_predicted_scores, results_dict_truth


@torch.no_grad()
def get_predictions(args, model, data_loader, device):
    # compute model predictions and truth
    results_dict_predicted_scores, results_dict_truth = get_model_scores(args, model, data_loader, device)
    return results_dict_predicted_scores, results_dict_truth


@torch.no_grad()
def evaluate(args, model, data_loader, device, ignore_macro=False, bias_correction_model=None):
    print('\nEvaluating...')

    # compute model predictions and truth
    results_dict_predicted_scores, results_dict_truth = get_model_scores(args, model, data_loader, device,
                                                                         bias_correction_model=bias_correction_model)

    # make metric dictionary
    metrics_dict = defaultdict(list)

    # compute auroc and map metrics and put results in dictionary
    for q_type in QType:

        if q_type == QType.spos or q_type == QType.spas or q_type == QType.spoo:
            # answer is a subject or object

            pred_results = results_dict_predicted_scores[q_type]
            truth_results = results_dict_truth[q_type]

            roc_score, mean_ap = compute_box_metrics(truth_results, pred_results)
            metrics_dict[q_type] = [roc_score, mean_ap]

            print('\nQType: ', q_type)
            print('ROC AUC ', roc_score)
            print('mAP ', mean_ap)

        elif q_type == QType.spop:
            # answer is a predicate

            preds = np.concatenate(results_dict_predicted_scores[q_type], axis=0)
            truth = np.concatenate(results_dict_truth[q_type], axis=0)

            roc_score_micro_full, mean_ap_micro_full = compute_metrics(truth, preds, metric_type='micro')

            # only true predicate classes (not has_attr)
            roc_score_micro_subset, mean_ap_micro_subset = compute_metrics(truth[:, 1:], preds[:, 1:],
                                                                           metric_type='micro')

            print('\nQType: ', q_type)
            print('ROC AUC (micro, full) ', roc_score_micro_full)
            print('mAP (micro, full) ', mean_ap_micro_full)

            if not ignore_macro:
                roc_score_macro_subset, mean_ap_macro_subset = compute_metrics(truth[:, 1:], preds[:, 1:],
                                                                               metric_type='macro')
                roc_score_macro_weight, mean_ap_macro_weight = compute_metrics(truth[:, 1:], preds[:, 1:],
                                                                               metric_type='weighted')

                metrics_dict[q_type] = [roc_score_micro_full, mean_ap_micro_full, roc_score_micro_subset,
                                        mean_ap_micro_subset, roc_score_macro_subset, mean_ap_macro_subset,
                                        roc_score_macro_weight, mean_ap_macro_weight]

                print('ROC AUC (micro, subset) ', roc_score_micro_subset)
                print('mAP (micro, subset) ', mean_ap_micro_subset)
                print('ROC AUC (macro, subset) ', roc_score_macro_subset)
                print('mAP (macro, subset) ', mean_ap_macro_subset)
                print('ROC AUC (macro, weight) ', roc_score_macro_weight)
                print('mAP (macro, weight) ', mean_ap_macro_weight)
            else:
                metrics_dict[q_type] = [roc_score_micro_full, mean_ap_micro_full, roc_score_micro_subset,
                                        mean_ap_micro_subset]

        elif q_type == QType.spap:
            preds = np.concatenate(results_dict_predicted_scores[q_type], axis=0)
            truth = np.concatenate(results_dict_truth[q_type], axis=0)

            roc_score, mean_ap = compute_metrics(truth, preds, metric_type='micro')
            metrics_dict[q_type] = [roc_score, mean_ap]

            print('\nQType: ', q_type)
            print('ROC AUC ', roc_score)
            print('mAP ', mean_ap)

        elif q_type == QType.spaa:
            # answer is an attribute
            preds = np.concatenate(results_dict_predicted_scores[q_type], axis=0)
            truth = np.concatenate(results_dict_truth[q_type], axis=0)
            preds = torch.from_numpy(preds)
            truth = torch.from_numpy(truth)

            roc_score_micro, mean_ap_micro = compute_metrics(truth, preds, metric_type='micro')

            print('\nQType: ', q_type)
            print('ROC AUC (micro) ', roc_score_micro)
            print('mAP (micro) ', mean_ap_micro)

            if not ignore_macro:
                roc_score_macro, mean_ap_macro = compute_metrics(truth, preds, metric_type='macro')
                roc_score_weight, mean_ap_weight = compute_metrics(truth, preds, metric_type='weighted')
                metrics_dict[q_type] = [roc_score_micro, mean_ap_micro, roc_score_macro, mean_ap_macro]

                print('ROC AUC (macro) ', roc_score_macro)
                print('mAP (macro) ', mean_ap_macro)
                print('ROC AUC (weight) ', roc_score_weight)
                print('mAP (weight) ', mean_ap_weight)
            else:
                metrics_dict[q_type] = [roc_score_micro, mean_ap_micro]
        else:
            raise NotImplementedError

    return metrics_dict


def evaluate_all_test_loaders(args, inc, epochs, model, test_loader, head_test_loader, tail_test_loader,
                              bias_correction_model, device, prefix=None):
    # evaluate model after current increment
    curr_results_dict = evaluate(args, model, test_loader, device, bias_correction_model=bias_correction_model)
    if epochs is not None:
        curr_results_dict['epochs'] = epochs

    # save results to json
    results_fname = 'full_results_increment_%d.json' % inc
    if prefix is not None:
        results_fname = prefix + results_fname
    json.dump(curr_results_dict, open(os.path.join(args.output_dir, results_fname), 'w'))

    if head_test_loader is not None and tail_test_loader is not None:
        curr_results_dict_tail = evaluate(args, model, tail_test_loader, device, ignore_macro=True,
                                          bias_correction_model=bias_correction_model)
        curr_results_dict_head = evaluate(args, model, head_test_loader, device, ignore_macro=True,
                                          bias_correction_model=bias_correction_model)

        # save results to json
        tail_results_fname = 'tail_results_increment_%d.json' % inc
        head_results_fname = 'head_results_increment_%d.json' % inc
        if prefix is not None:
            tail_results_fname = prefix + tail_results_fname
            head_results_fname = prefix + head_results_fname
        json.dump(curr_results_dict_tail,
                  open(os.path.join(args.output_dir, tail_results_fname), 'w'))
        json.dump(curr_results_dict_head,
                  open(os.path.join(args.output_dir, head_results_fname), 'w'))

    return curr_results_dict
