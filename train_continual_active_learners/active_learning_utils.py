import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_continual_active_learners.triple_completion_utils import compute_split_sizes, compute_dist_scores, \
    create_questions_for_batch
from train_continual_active_learners.vg_qtype_dset import collate_fn_qtype
from train_continual_active_learners.vg_qtype_dset_loader_utils import get_head_mid_tail_class_ixs
from utils import QType


@torch.no_grad()
def get_al_value_from_proba_vector(probas, method, device, eps=1e-12):
    if 'confidence' in method:
        probas = torch.softmax(probas, dim=1)
        values = torch.max(probas)
    elif 'margin' in method:
        probas = torch.softmax(probas, dim=1)
        top2 = torch.topk(probas, 2)[0][0]
        values = top2[0] - top2[1]
    elif 'entropy' in method:
        probas = torch.softmax(probas, dim=1)
        log_probs = probas * torch.log2(probas + eps)  # multiply each probability by its base 2 log
        values = - torch.mean(log_probs, dim=1)
    else:
        raise NotImplementedError
    return values


def weighted_choice(proba, ixs, num_samples, min_val=True, replace=False, shift_min=True, eps=1e-7):
    proba = np.array(proba)
    if shift_min:
        proba = proba + (1 - np.min(proba))  # shift smallest probability to 1
        if min_val:
            proba = 1 / (proba + eps)
        else:
            proba = proba
        proba = proba / np.linalg.norm(proba, ord=1)  # sum to 1
    else:
        if min_val:
            proba = 1 / (proba + eps)
        else:
            proba = proba
        proba = proba / np.linalg.norm(proba, ord=1)  # sum to 1

    np.random.seed(666)
    return np.random.choice(ixs, size=min(num_samples, len(proba)), replace=replace, p=proba)


def get_active_learning_data_loader(args, dataset, batch_size, drop_last=True, sampler=None):
    if sampler is None:
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': True,
            'drop_last': drop_last,
            'collate_fn': collate_fn_qtype,
        }
    else:
        # sampler option is mutually exclusive with shuffle
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': args.loader_num_workers,
            'drop_last': drop_last,
            'collate_fn': collate_fn_qtype,
            'sampler': sampler,
        }
    train_loader = DataLoader(dataset, **loader_kwargs)
    return train_loader


def get_active_learning_model_scores(args, method, model, data_loader, device):
    # setup model
    model = model.to(device)
    model.eval()

    # keys are q_types and value is list of distance_scores/one_hot_encoded_truth
    values_dict = {}
    qtype_dict = {}
    orig_index_dict = {}
    class_ix_dict = {}
    class_uncertainty_scores_dict = {}

    # compute embeddings for all one-hot combinations of predicates and attributes
    pred_batch = torch.arange(args.num_predicate_categories).to(device)
    attr_batch = torch.arange(args.num_attribute_categories).to(device)
    predicate_embeddings = model.compute_predicate_embeddings(pred_batch, target=True)
    attr_embeddings = model.compute_attribute_embeddings(attr_batch, target=True)

    # get class indices for head and tail classes
    head_class_ix, tail_class_ix = get_head_mid_tail_class_ixs(args, merge_mid_and_tail=True)
    attribute_probas, predicate_probas = make_dictionary_of_tail_probabilities(args, head_class_ix, tail_class_ix)

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        original_indices = batch['original_index']
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

        if method in ['random', 'tail', 'tail_uniform_class', 'tail_count_proba']:
            # don't need network outputs, so just compute active learning scores directly
            compute_active_learning_raw_scores(args, model, method, ques_target, predicate_labels, attr_labels,
                                               values_dict, qtype_dict, orig_index_dict, class_ix_dict,
                                               class_uncertainty_scores_dict, indices, original_indices,
                                               predicate_probas, attribute_probas, device)

        else:
            # compute network predictions to questions and target predictions
            predicted_features = model(bbox_batch, predicate_labels, attr_labels, ques_triples, device)

            # get features for all boxes in the batch
            curr_batch_box_features = model.compute_bbox_embeddings(bbox_batch.to(device), target=True)

            # compute model scores
            compute_active_learning_network_scores(args, method, ques_triples, ques_target, predicted_features,
                                                   curr_batch_box_features, predicate_labels, attr_labels,
                                                   predicate_embeddings, attr_embeddings,
                                                   box_feature_ids_to_subject_box_ids, values_dict, qtype_dict,
                                                   orig_index_dict, class_ix_dict, indices, original_indices,
                                                   predicate_probas, attribute_probas, device)

    return values_dict, qtype_dict, orig_index_dict, class_ix_dict, class_uncertainty_scores_dict


def compute_active_learning_network_scores(args, method, ques_triples, ques_target, predicted_features,
                                           curr_batch_box_features, predicate_batch, attribute_batch,
                                           predicate_embeddings, attr_embeddings, box_feature_ids_to_subject_box_ids,
                                           values_dict, qtype_dict, orig_index_dict, class_ix_dict, index, orig_index,
                                           predicate_probas, attribute_probas, device):
    # whether a small probability is "best" for active selection
    if method in ['confidence', 'margin', 'confidence_no_head', 'margin_no_head']:
        min_val = True
    else:
        min_val = False

    for q_type in QType:
        q_ix = np.where(ques_target[:, 0] == q_type)[0]
        if len(q_ix) == 0:
            # no questions of this type, continue
            continue
        curr_target = ques_target[q_ix]
        curr_triples = ques_triples[q_ix]
        curr_pred_feats = predicted_features[q_ix]
        curr_predicates = predicate_batch[q_ix]
        curr_attributes = attribute_batch[q_ix]
        curr_qid = index[q_ix]
        curr_orig_id = orig_index[q_ix]

        if q_type == QType.spos or q_type == QType.spas:
            # answer is a box

            # get dists for each feature based on box features in same scene graph. also return idx of true box feature
            for j in range(len(curr_target)):

                subject_box_id = curr_target[:, 1][j]
                curr_pred_feat = curr_pred_feats[j].unsqueeze(0)
                idxs = torch.where(box_feature_ids_to_subject_box_ids == subject_box_id)[0]
                boxes_in_graph = curr_batch_box_features[idxs]
                probas = compute_dist_scores(args.embed_dist_metric, boxes_in_graph, curr_pred_feat)
                value = get_al_value_from_proba_vector(probas, method, device, eps=1e-12)
                value = value.cpu().item()

                if q_type == QType.spos:
                    curr_cls = 'p_' + str(curr_predicates[j].item())
                elif q_type == QType.spas:
                    curr_cls = 'a_' + str(curr_attributes[j].item())

                if 'no_head' in method and q_type == QType.spos:
                    curr_pred = curr_predicates[j]
                    if predicate_probas[curr_pred.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0
                elif 'no_head' in method and q_type == QType.spas:
                    curr_attr = curr_attributes[j]
                    if attribute_probas[curr_attr.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls

        elif q_type == QType.spoo:
            # answer is a box

            # get dists for each feature based on box features in same scene graph. also return idx of true box feature
            for j in range(len(curr_triples)):

                subject_box_id = curr_triples[:, 1][j]
                curr_pred_feat = curr_pred_feats[j].unsqueeze(0)
                idxs = torch.where(box_feature_ids_to_subject_box_ids == subject_box_id)[0]
                boxes_in_graph = curr_batch_box_features[idxs]
                probas = compute_dist_scores(args.embed_dist_metric, boxes_in_graph, curr_pred_feat)
                value = get_al_value_from_proba_vector(probas, method, device, eps=1e-12)
                value = value.cpu().item()

                curr_cls = 'p_' + str(curr_predicates[j].item())

                if 'no_head' in method:
                    curr_pred = curr_predicates[j]
                    if predicate_probas[curr_pred.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls

        elif q_type == QType.spop or q_type == QType.spap:
            # answer is a predicate
            dists = compute_dist_scores(args.embed_dist_metric, predicate_embeddings, curr_pred_feats)

            for j, dist in enumerate(dists):

                probas = dist.unsqueeze(0)
                value = get_al_value_from_proba_vector(probas, method, device, eps=1e-12)
                value = value.cpu().item()

                if q_type == QType.spop:
                    curr_cls = 'p_' + str(curr_predicates[j].item())
                elif q_type == QType.spap:
                    curr_cls = 'a_' + str(curr_attributes[j].item())

                if 'no_head' in method and q_type == QType.spop:
                    curr_pred = curr_predicates[j]
                    if predicate_probas[curr_pred.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0
                elif 'no_head' in method and q_type == QType.spap:
                    curr_attr = curr_attributes[j]
                    if attribute_probas[curr_attr.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls

        elif q_type == QType.spaa:
            # answer is an attribute
            dists = compute_dist_scores(args.embed_dist_metric, attr_embeddings, curr_pred_feats)

            for j, dist in enumerate(dists):

                probas = dist.unsqueeze(0)
                value = get_al_value_from_proba_vector(probas, method, device, eps=1e-12)
                value = value.cpu().item()
                curr_cls = 'a_' + str(curr_attributes[j].item())

                if 'no_head' in method:
                    curr_attr = curr_attributes[j]
                    if attribute_probas[curr_attr.item()] == args.tail_head_probability:
                        # zero probability for head class
                        if min_val:
                            value = 10000000
                        else:
                            value = 0

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls

        else:
            raise NotImplementedError


def compute_active_learning_raw_scores(args, model, method, ques_target, predicate_batch, attribute_batch,
                                       values_dict, qtype_dict, orig_index_dict, class_ix_dict, class_uncertainty_dict,
                                       index, orig_index, predicate_probas, attribute_probas, device):
    for q_type in QType:
        q_ix = np.where(ques_target[:, 0] == q_type)[0]
        if len(q_ix) == 0:
            # no questions of this type, continue
            continue
        curr_target = ques_target[q_ix]
        curr_predicates = predicate_batch[q_ix]
        curr_attributes = attribute_batch[q_ix]
        curr_qid = index[q_ix]
        curr_orig_id = orig_index[q_ix]

        if q_type == QType.spos or q_type == QType.spop or q_type == QType.spoo:

            for j in range(len(curr_target)):
                curr_pred = curr_predicates[j]
                curr_cls = 'p_' + str(curr_pred.item())

                if 'tail' in method:
                    if args.tail_seen_distribution:
                        num_seen = model.visited_predicate_counts[curr_pred.item()]
                        value = 1 / (1 + num_seen)
                    else:
                        value = predicate_probas[curr_pred.item()]
                else:
                    # random
                    value = random.random()

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls
                class_uncertainty_dict[curr_cls] = value  # value is assigned to a class

        elif q_type == QType.spas or q_type == QType.spap or q_type == QType.spaa:

            for j in range(len(curr_target)):
                curr_attr = curr_attributes[j]
                curr_cls = 'a_' + str(curr_attr.item())

                if 'tail' in method:
                    if args.tail_seen_distribution:
                        num_seen = model.visited_attribute_counts[curr_attr.item()]
                        value = 1 / (1 + num_seen)
                    else:
                        value = attribute_probas[curr_attr.item()]
                else:
                    # random
                    value = random.random()

                values_dict[curr_qid[j]] = value
                qtype_dict[curr_qid[j]] = q_type
                orig_index_dict[curr_qid[j]] = curr_orig_id[j]
                class_ix_dict[curr_qid[j]] = curr_cls
                class_uncertainty_dict[curr_cls] = value  # value is assigned to a class

        else:
            raise NotImplementedError


def make_dictionary_of_tail_probabilities(args, head_class_ix, tail_class_ix):
    # this is faster than checking if a predicate or attribute is in a specific set of categories each iteration
    attributes = {}
    predicates = {}
    for ix in head_class_ix['attributes']:
        attributes[ix] = args.tail_head_probability
    for ix in head_class_ix['predicates']:
        predicates[ix] = args.tail_head_probability
    for ix in tail_class_ix['attributes']:
        attributes[ix] = args.tail_tail_probability
    for ix in tail_class_ix['predicates']:
        predicates[ix] = args.tail_tail_probability
    return attributes, predicates


def perform_active_sampling(sampling_type, value_dict, qtype_dict, orig_index_dict, class_ix_dict,
                            class_uncertainty_scores_dict, al_method, num_samples, num_questions=6):
    values = np.array(list(value_dict.values()))
    keys = np.array(list(value_dict.keys()))
    orig_indices = np.array(list(orig_index_dict.values()))
    min_val = al_method in ['confidence', 'margin', 'confidence_no_head', 'margin_no_head']
    if al_method in ['tail_uniform_class', 'tail_count_proba']:
        # assumes balanced selection among question types with probabilistic random sampling
        chosen_ixs = active_sampling_over_classes(class_ix_dict, class_uncertainty_scores_dict, qtype_dict, num_samples,
                                                  num_questions)
    else:
        if sampling_type == 'probabilistic':
            # use active learning scores as probabilities for choosing samples
            chosen_ixs = weighted_choice(values, np.arange(len(value_dict)), num_samples, min_val)
        elif sampling_type == 'rank' and al_method != 'random':
            # use raw active learning scores to choose samples
            if min_val:
                # desire smallest values
                chosen_ixs = np.argpartition(values, num_samples)[:num_samples]
            else:
                # desire largest values
                chosen_ixs = np.argpartition(values, -num_samples)[-num_samples:]
        elif 'balanced' in sampling_type:
            if 'no_head' in al_method:
                shift = False
            else:
                shift = True
            chosen_ixs = []
            sizes = compute_split_sizes(num_samples, num_questions)
            q_list = np.array(list(qtype_dict.values()))
            for i, s in enumerate(sizes):
                ix = np.where(q_list == i)[0]
                curr_vals = values[ix]
                if 'rank' in sampling_type:
                    # shuffle so that all samples from same class aren't next to each other
                    arr_ixs = np.arange(len(ix))
                    np.random.shuffle(arr_ixs)
                    ix = ix[arr_ixs]
                    curr_vals = curr_vals[arr_ixs]
                    if min_val:
                        # desire smallest values
                        chosen_ixs += list(ix[np.argpartition(curr_vals, s)[:s]])
                    else:
                        # desire largest values
                        chosen_ixs += list(ix[np.argpartition(curr_vals, -s)[-s:]])
                else:
                    chosen_ixs += list(weighted_choice(curr_vals, ix, s, min_val=min_val, shift_min=shift))
            chosen_ixs = np.array(chosen_ixs)
        else:
            raise NotImplementedError

    assert len(chosen_ixs) == num_samples

    keys = keys[chosen_ixs]
    orig_indices = orig_indices[chosen_ixs]
    return keys, orig_indices


def active_sampling_over_classes(class_ix_dict, class_uncertainty_scores_dict, qtype_dict, num_samples, num_questions):
    print('\nSelecting samples in class-wise fashion...')
    chosen_ixs = []
    sizes = compute_split_sizes(num_samples, num_questions)
    q_list = np.array(list(qtype_dict.values()))
    class_ixs = np.array(list(class_ix_dict.values()))
    for i, s in enumerate(sizes):
        curr_chosen = 0
        ix = np.where(q_list == i)[0]
        curr_cls = class_ixs[ix]  # get classes represented in current qtype
        unique_curr_cls = np.unique(curr_cls)

        # make list of class scores to match unique class order
        cls_scores = []
        for cls in unique_curr_cls:
            cls_scores.append(class_uncertainty_scores_dict[cls])

        # remove anything with 0 probability
        cls_score_bool = np.array(cls_scores, dtype=np.bool)
        unique_curr_cls = unique_curr_cls[cls_score_bool]
        cls_scores = np.array(cls_scores)[cls_score_bool]

        # select number of samples per class based on probabilities
        selected_classes = np.random.choice(unique_curr_cls, size=s, replace=True, p=cls_scores / sum(cls_scores))
        unique_cls, counts = np.unique(selected_classes, return_counts=True)

        # iterate over classes and select samples
        for cls, count in zip(unique_cls, counts):
            # select samples from the class at random
            curr_ix = np.where(curr_cls == cls)[0]
            select = np.random.choice(curr_ix, min(count, len(curr_ix)), replace=False)
            chosen_ixs.extend(list(ix[select]))  # get index with respect to large list
            curr_chosen += len(select)

        if curr_chosen != s:
            # not enough samples, so let's add some
            num_needed = s - curr_chosen
            chosen_ixs.extend(list(np.random.choice(ix, num_needed,
                                                    replace=False)))  # should only be for really tiny classes, so let's just randomly grab some

    chosen_ixs = np.array(chosen_ixs).flatten()
    return chosen_ixs


def choose_active_learning_samples(args, model, al_method, unlabeled_loader, device):
    if args.active_learning_method == 'random' and 'balanced' not in args.sampling_type:
        '\nUsing random sampling for active learning...'
        num_samples = len(unlabeled_loader.dataset)
        chosen_ixs = weighted_choice(np.ones(num_samples), np.arange(num_samples),
                                     args.num_active_learning_samples, True)
        orig_chosen_ixs = np.array(())  # TODO: populate this
    else:
        values_dict, qtype_dict, orig_index_dict, class_ix_dict, class_uncertainty_scores_dict = get_active_learning_model_scores(
            args, al_method, model, unlabeled_loader, device)
        chosen_ixs, orig_chosen_ixs = perform_active_sampling(args.sampling_type, values_dict, qtype_dict,
                                                              orig_index_dict, class_ix_dict,
                                                              class_uncertainty_scores_dict, al_method,
                                                              args.num_active_learning_samples)

    unlabel_ixs = np.setdiff1d(np.arange(len(unlabeled_loader.dataset)), chosen_ixs)
    active_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, indices=chosen_ixs)
    unlabel_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, indices=unlabel_ixs)
    return active_dataset, unlabel_dataset, chosen_ixs, unlabel_ixs, orig_chosen_ixs
