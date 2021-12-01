import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils import QType, CMA


def get_scores_and_truth(args, model, batch, device):
    # grab items from batch
    question_id = batch['question_id']
    box_features = batch['box_features']
    subject_box_id = batch['subject_box_id']
    object_box_id = batch['object_box_id']
    predicate_labels = batch['predicates']
    attr_labels = batch['attributes']

    # setup data and feed through model
    bbox_batch = box_features.to(device)  # num_boxes x bbox_feat_size
    ques_triples, ques_targets = create_questions_for_batch(args, question_id, subject_box_id, object_box_id)
    predicate_labels = predicate_labels.to(device)
    attr_labels = attr_labels.to(device)

    # compute network predictions to questions and target predictions
    predicted_features = model(bbox_batch, predicate_labels, attr_labels, ques_triples, device)
    target_features = model.get_target_features(bbox_batch, predicate_labels, attr_labels, ques_targets, device)

    # compute inner product of predicted features with target features (truth on diagonal)
    scores = compute_dist_scores(args.embed_dist_metric, target_features, predicted_features)
    truth = torch.arange(len(scores)).to(device)
    scores = scores.float()

    return scores, truth


def compute_split_sizes(num_samples, k):
    # compute train and validation split sizes for cross-validation
    size = num_samples // k
    leftover = num_samples % k
    sizes = np.ones(k, dtype=np.int64) * size
    if leftover != 0:
        # distribute leftover samples
        for i in range(leftover):
            sizes[i % k] += 1

    assert sum(sizes) == num_samples
    return sizes


def compute_dist_scores(dist_metric, A, B):
    """
    Given a matrix of points A, return the scores (negative distances) of points in A to B using args.embed_dist_metric distance.
    :param A: N x d matrix of points
    :param B: M x d matrix of points for predictions
    :return: M x N matrix of distances
    """
    M, d = B.shape
    A = A.double()
    B = B.double()
    if dist_metric == 'l2':
        B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
        square_sub = torch.mul(A - B, A - B)  # square all elements
        dist = -torch.sum(square_sub, dim=2)  # negate distances for scores
    elif dist_metric == 'inner_prod':
        dist = torch.mm(B, torch.t(A))
    elif dist_metric == 'cosine':
        B = F.normalize(B, p=2, dim=1)
        A = F.normalize(A, p=2, dim=1)
        dist = torch.mm(B, torch.t(A))
    else:
        raise NotImplementedError
    return dist.float()


def create_questions_for_batch(args, question_types, subjects, objects):
    # form question and answer batches for training and evaluation

    num_questions = len(question_types)
    triple_questions = torch.empty((num_questions, 3), dtype=torch.long)
    target_questions = torch.empty((num_questions, 2), dtype=torch.long)
    for q_type in QType:
        curr_ixs = torch.where(question_types == q_type)[0]
        if len(curr_ixs) == 0:
            # no questions of this type, continue
            continue
        subj = subjects[curr_ixs]
        obj = objects[curr_ixs]
        if q_type == QType.spos:
            ix0 = curr_ixs
            ix1 = obj
            ix2 = subj
        elif q_type == QType.spas:
            ix0 = curr_ixs
            ix1 = curr_ixs
            ix2 = subj
        elif q_type == QType.spop:
            ix0 = subj
            ix1 = obj
            ix2 = curr_ixs
        elif q_type == QType.spap:
            ix0 = subj
            ix1 = curr_ixs
            ix2 = curr_ixs
        elif q_type == QType.spoo:
            ix0 = subj
            ix1 = curr_ixs
            ix2 = obj
        elif q_type == QType.spaa:
            ix0 = subj
            ix1 = curr_ixs
            ix2 = curr_ixs
        else:
            raise NotImplementedError
        triple_questions[curr_ixs, 0] = q_type
        triple_questions[curr_ixs, 1] = ix0
        triple_questions[curr_ixs, 2] = ix1
        target_questions[curr_ixs, 0] = q_type
        target_questions[curr_ixs, 1] = ix2

    return triple_questions, target_questions


def find_ix_a_in_b(a, b):
    ix = (a.unsqueeze(1) == b).nonzero(as_tuple=True)[0]
    return ix


def find_set_diff(a, b):
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference


def get_distribution(data_loader, model, add_to_model=True):
    qtype_dict = defaultdict(int)
    attr_dict = defaultdict(int)
    pred_dict = defaultdict(int)

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        question_id = batch['question_id']
        predicate_labels = batch['predicates']
        attr_labels = batch['attributes']

        # determine which questions contain attributes
        attr_q = torch.cat([torch.where(question_id == QType.spap)[0], torch.where(question_id == QType.spas)[0],
                            torch.where(question_id == QType.spaa)[0]])

        # store predicate counts in dictionary
        for pred in predicate_labels:
            pred_dict[pred.item()] += 1
            if add_to_model:
                model.visited_predicate_counts[pred.item()] += 1

        # store attribute counts in dictionary
        for attr in attr_labels[attr_q]:
            attr_dict[attr.item()] += 1
            if add_to_model:
                model.visited_attribute_counts[attr.item()] += 1

        for q in question_id:
            qtype_dict[q.item()] += 1
            if add_to_model:
                model.visited_qtype_counts[q.item()] += 1

    return qtype_dict, attr_dict, pred_dict


def save_sample_distributions(args, loader, model, file_name):
    # save out attribute class, predicate class, and q-type distributions
    q_dist, a_dist, p_dist = get_distribution(loader, model)

    d = {}
    d['qtype'] = q_dist
    d['attributes'] = a_dist
    d['predicates'] = p_dist

    # save dictionary out to json files
    save_file = os.path.join(args.output_dir, file_name)
    print('\nSaving train sample distribution out to: %s' % save_file)
    with open(save_file, 'w', encoding="utf8") as f:
        json.dump(d, f)
    return d


@torch.no_grad()
def compute_val_loss(args, model, val_loader, criterion, device):
    model.eval()
    val_loss = CMA()
    for i, batch in enumerate(val_loader):
        scores, truth = get_scores_and_truth(args, model, batch, device)
        loss = criterion(scores, truth)
        loss = torch.mean(loss)
        val_loss.update(loss.item())
    model.train()
    return val_loss.avg
