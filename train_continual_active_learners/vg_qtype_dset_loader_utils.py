from vg_qtype_dset import SceneGraphQuestionDataset, collate_fn_qtype

import json
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import torch


class ReindexDataset(Dataset):
    """
    Dataset class so we can remap indices back to 0 instead of keeping original subset index references. This is useful
    when splitting our dataset up based on active learning scores.
    """

    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        target = self.subset[index]
        target['index'] = index
        # return local index
        return target

    def __len__(self):
        return len(self.subset)


def filter_pre_train_classes(args, attr_class_ix, pred_class_ix, train_dset, num_samples=-1):
    orig_len = len(train_dset)
    np.random.seed(args.class_permutation_seed)

    # load attribute and predicate class dictionaries (key=class, value=list(indices))
    with open(args.train_attribute_dict, 'r') as f:
        train_attr_dict = json.load(f)

    with open(args.train_predicate_dict, 'r') as f:
        train_pred_dict = json.load(f)

    # filter box_ids based on desired attribute and predicate categories
    attr_id_list = []
    for cls_ix in attr_class_ix:
        # randomly sample num_samples from each class
        attr_qs = train_attr_dict[str(cls_ix)]
        if num_samples != -1:
            samples_ix = np.random.choice(np.arange(len(attr_qs)), num_samples, replace=False)
            attr_qs = list(np.array(attr_qs)[samples_ix])
        attr_id_list.extend(attr_qs)
    pred_id_list = []
    for cls_ix in pred_class_ix:

        # skip "has_attr" class
        if cls_ix == 0:
            continue

        # randomly sample num_samples from each class
        pred_qs = train_pred_dict[str(cls_ix)]
        if num_samples != -1:
            samples_ix = np.random.choice(np.arange(len(pred_qs)), num_samples, replace=False)
            pred_qs = list(np.array(pred_qs)[samples_ix])
        pred_id_list.extend(pred_qs)

    attr_id_list = np.unique(np.array(attr_id_list))
    pred_id_list = np.unique(np.array(pred_id_list))

    keep_ix = np.unique(np.concatenate([attr_id_list, pred_id_list]))
    total_ix = np.arange(orig_len)
    unlabeled_ix = np.setdiff1d(total_ix, keep_ix)

    pre_train_labeled_dset = torch.utils.data.Subset(train_dset, indices=keep_ix)
    unlabeled_train_dset = torch.utils.data.Subset(train_dset, indices=unlabeled_ix)
    unlabeled_train_dset = ReindexDataset(unlabeled_train_dset)  # zero-based indexing for active learning
    train_dset = pre_train_labeled_dset
    return train_dset, pre_train_labeled_dset, unlabeled_train_dset


def build_dsets(args):
    start = time.time()
    print("building unpaired %s dataset" % args.dataset)

    np.random.seed(args.class_permutation_seed)

    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    vocab['attribute_name_to_idx']['__attr__'] = -1
    vocab['attribute_idx_to_name'] += ['__attr__']
    dset_kwargs = {
        'vocab': vocab,
        'dset_type': 'train',
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'box_feature_h5': args.train_box_feature_h5,
        'box_h5': args.train_box_h5,
        'question_h5': args.train_question_h5,
        'triple_h5': args.train_triple_h5,
        'attribute_h5': args.train_attribute_h5,
        'load_img': args.load_images
    }
    train_dset = SceneGraphQuestionDataset(**dset_kwargs)
    class_ix_dict = {}

    dset_kwargs['dset_type'] = 'val'
    dset_kwargs['h5_path'] = args.val_h5
    dset_kwargs['box_feature_h5'] = args.val_box_feature_h5
    dset_kwargs['box_h5'] = args.val_box_h5
    dset_kwargs['question_h5'] = args.val_question_h5
    dset_kwargs['triple_h5'] = args.val_triple_h5
    dset_kwargs['attribute_h5'] = args.val_attribute_h5
    val_dset = SceneGraphQuestionDataset(**dset_kwargs)

    dset_kwargs['dset_type'] = 'test'
    dset_kwargs['h5_path'] = args.test_h5
    dset_kwargs['box_feature_h5'] = args.test_box_feature_h5
    dset_kwargs['box_h5'] = args.test_box_h5
    dset_kwargs['question_h5'] = args.test_question_h5
    dset_kwargs['triple_h5'] = args.test_triple_h5
    dset_kwargs['attribute_h5'] = args.test_attribute_h5
    test_dset = SceneGraphQuestionDataset(**dset_kwargs)

    if args.num_head_samples_per_class is not None:
        # use balanced number of samples from all head classes
        class_ix_dict, mid_class_ix, tail_class_ix = get_head_mid_tail_class_ixs(args)
        train_dset, pre_train_labeled_dset, unlabeled_train_dset = filter_pre_train_classes(args,
                                                                                            class_ix_dict[
                                                                                                'attributes'],
                                                                                            class_ix_dict[
                                                                                                'predicates'],
                                                                                            train_dset,
                                                                                            num_samples=args.num_head_samples_per_class)

    else:
        # no active learning samples, set active learning datasets to None
        pre_train_labeled_dset = None
        unlabeled_train_dset = None

    print('\nObtained data in time: ', (time.time() - start))
    print('Len Train Dset: ', len(train_dset))
    print('Len Val Dset: ', len(val_dset))
    print('Len Test Dset: ', len(test_dset))
    if pre_train_labeled_dset is not None:
        print('Len Pre-Train Labeled Dset: ', len(pre_train_labeled_dset))
    if unlabeled_train_dset is not None:
        print('Len Unlabeled Train Dset: ', len(unlabeled_train_dset))

    return vocab, train_dset, unlabeled_train_dset, pre_train_labeled_dset, val_dset, test_dset, class_ix_dict


def get_class_specific_test_loader(args, attr_dict_file, pred_dict_file, class_ix, dset):
    with open(attr_dict_file, 'r') as f:
        attr_dict = json.load(f)
    with open(pred_dict_file, 'r') as f:
        pred_dict = json.load(f)

    attr_class_ix = class_ix['attributes']
    pred_class_ix = class_ix['predicates']
    attr_id_list = []
    for cls_ix in attr_class_ix:
        attr_id_list.extend(attr_dict[str(cls_ix)])
    pred_id_list = []
    for cls_ix in pred_class_ix:
        pred_id_list.extend(pred_dict[str(cls_ix)])
    attr_id_list = np.unique(np.array(attr_id_list))
    pred_id_list = np.unique(np.array(pred_id_list))
    keep_ix = np.unique(np.concatenate([attr_id_list, pred_id_list]))
    filtered_dset = torch.utils.data.Subset(dset, indices=keep_ix)
    filtered_loader = get_generic_data_loader(args, filtered_dset, batch_size=args.batch_size, shuffle=False)
    return filtered_loader


def get_generic_data_loader(args, dset, batch_size, shuffle=True, drop_last=False, sampler=None):
    collate_fn = collate_fn_qtype

    # sampler option is mutually exclusive with shuffle
    if sampler is None:
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'drop_last': drop_last,
            # 'pin_memory': True,
        }
    else:
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': args.loader_num_workers,
            'collate_fn': collate_fn,
            'drop_last': drop_last,
            'sampler': sampler,
            # 'pin_memory': True,
        }
    loader = DataLoader(dset, **loader_kwargs)
    return loader


def build_loaders(args, shuffle=True, drop_last=False):
    print(args.dataset)
    vocab, train_dset, unlabeled_train_dset, pre_train_labeled_dset, val_dset, test_dset, class_ix_dict = build_dsets(
        args)
    collate_fn = collate_fn_qtype

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'collate_fn': collate_fn,
        # 'pin_memory': True,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    pre_train_labeled_loader = DataLoader(pre_train_labeled_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    unlabeled_train_loader = DataLoader(unlabeled_train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    loader_kwargs['drop_last'] = False
    loader_kwargs['batch_size'] = args.test_batch_size
    val_loader = DataLoader(val_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    loader_kwargs['batch_size'] = args.test_batch_size
    test_loader = DataLoader(test_dset, **loader_kwargs)

    return vocab, train_loader, unlabeled_train_loader, pre_train_labeled_loader, val_loader, test_loader, class_ix_dict


def get_head_mid_tail_ixs(d, high_cutoff, low_cutoff, merge_mid_and_tail=False):
    vals = np.array(list(d.values()))
    keys = np.array(list(d.keys()))

    head_ix = np.where(vals > high_cutoff)[0]
    head_ix = keys[head_ix]

    ix1 = np.where(vals <= high_cutoff)[0]
    ix2 = np.where(vals > low_cutoff)[0]
    mid_ix = np.intersect1d(ix1, ix2)
    mid_ix = keys[mid_ix]

    tail_ix = np.where(vals <= low_cutoff)[0]
    tail_ix = keys[tail_ix]

    if merge_mid_and_tail:
        return head_ix, np.unique(np.concatenate([np.array(mid_ix), np.array(tail_ix)]))
    else:
        return head_ix, mid_ix, tail_ix


def get_head_mid_tail_class_ixs(args, low_cutoff_pred=3000, high_cutoff_pred=15000, low_cutoff_attr=5000,
                                high_cutoff_attr=10000, merge_mid_and_tail=False):
    # filter train data to find tail classes
    with open(args.train_attribute_dict, 'r') as f:
        train_attr_dict = json.load(f)

    with open(args.train_predicate_dict, 'r') as f:
        train_pred_dict = json.load(f)

    # make count dictionaries of frequency
    count_dict_attr = {}
    for k, v in train_attr_dict.items():
        count_dict_attr[int(k)] = len(v)

    count_dict_pred = {}
    for k, v in train_pred_dict.items():
        count_dict_pred[int(k)] = len(v)

    # sort counts by frequency
    sort_attr = {k: v for k, v in sorted(count_dict_attr.items(), key=lambda item: item[1])}
    sort_pred = {k: v for k, v in sorted(count_dict_pred.items(), key=lambda item: item[1])}

    # get head, mid, and tail class ixs
    if merge_mid_and_tail:
        head_class_ix = {}
        tail_class_ix = {}
        head_class_ix['attributes'], tail_class_ix['attributes'] = get_head_mid_tail_ixs(
            sort_attr, high_cutoff_attr, low_cutoff_attr, merge_mid_and_tail=True)
        head_class_ix['predicates'], tail_class_ix['predicates'] = get_head_mid_tail_ixs(
            sort_pred, high_cutoff_pred, low_cutoff_pred, merge_mid_and_tail=True)

        # remove has_attr predicate
        head_class_ix['predicates'] = head_class_ix['predicates'][np.where(head_class_ix['predicates'] != 0)[0]]

        return head_class_ix, tail_class_ix
    else:
        head_class_ix = {}
        mid_class_ix = {}
        tail_class_ix = {}
        head_class_ix['attributes'], mid_class_ix['attributes'], tail_class_ix['attributes'] = get_head_mid_tail_ixs(
            sort_attr, high_cutoff_attr, low_cutoff_attr)
        head_class_ix['predicates'], mid_class_ix['predicates'], tail_class_ix['predicates'] = get_head_mid_tail_ixs(
            sort_pred, high_cutoff_pred, low_cutoff_pred)

        return head_class_ix, mid_class_ix, tail_class_ix


def get_head_and_tail_test_loaders(args, test_dset):
    # get filtered test loaders
    head_class_ix, tail_class_ix = get_head_mid_tail_class_ixs(args, merge_mid_and_tail=True)
    head_test_loader = get_class_specific_test_loader(args, args.test_attribute_dict, args.test_predicate_dict,
                                                      head_class_ix, test_dset)
    tail_test_loader = get_class_specific_test_loader(args, args.test_attribute_dict, args.test_predicate_dict,
                                                      tail_class_ix, test_dset)
    assert (len(head_test_loader.dataset) + len(tail_test_loader.dataset)) == len(test_dset)
    return head_test_loader, tail_test_loader
