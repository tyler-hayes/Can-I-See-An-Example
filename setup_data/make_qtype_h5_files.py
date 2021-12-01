import argparse, os
import h5py
from tqdm import tqdm
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vg_image_dset import SceneGraphNoPairsDataset, collate_fn_nopairs
from vg_box_dset import SceneGraphBoxDataset, collate_fn_boxes
from utils import QType, bool_flag
from config import META_DATA_DIR, DATA_DIR


def get_args_parser(add_help=False):
    parser = argparse.ArgumentParser(description='Attribute & Predicate Prediction', add_help=add_help)

    # Dataset options
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--shuffle_val', default=False, type=bool_flag)
    parser.add_argument('--loader_num_workers', default=8, type=int)
    parser.add_argument('--vg_image_dir', default=os.path.join(DATA_DIR, ''))
    parser.add_argument('--train_h5', default=os.path.join(META_DATA_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(META_DATA_DIR, 'val.h5'))
    parser.add_argument('--test_h5', default=os.path.join(META_DATA_DIR, 'test.h5'))
    parser.add_argument('--vocab_json', default=os.path.join(META_DATA_DIR, 'vocab.json'))
    parser.add_argument('--use_unnormalized_boxes', default=True, type=bool_flag)
    parser.add_argument('--box_dataset', default=True, type=bool_flag)

    # bounding box feature h5 files
    parser.add_argument('--train_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_train.h5'))
    parser.add_argument('--val_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_val.h5'))
    parser.add_argument('--test_box_h5', default=os.path.join(META_DATA_DIR, 'vg_box_rn50_features_test.h5'))

    # directory parameters
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num_train_samples', type=int, default=None)  # pre-train number of samples
    parser.add_argument('--num_attribute_pre_train_classes', type=int, default=None)
    parser.add_argument('--num_relationship_pre_train_classes', type=int, default=None)

    return parser


class RandomSplitReindex(Dataset):
    """
    Dataset class so we can remap indices back to 0 instead of keeping original subset index references.
    """

    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        target = self.subset[index]
        target['box_id'] = index
        # return local index
        return target

    def __len__(self):
        return len(self.subset)


def build_train_dsets(args):
    start = time.time()
    print("building unpaired %s dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    vocab['attribute_name_to_idx']['__attr__'] = -1
    vocab['attribute_idx_to_name'] += ['__attr__']
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'use_unnormalized_boxes': args.use_unnormalized_boxes,
        'box_h5': args.train_box_h5,
    }
    if args.box_dataset:
        dset_kwargs['filter_curr_graphs'] = args.filter_curr_graphs
        train_dset = SceneGraphBoxDataset(**dset_kwargs)
    else:
        train_dset = SceneGraphNoPairsDataset(**dset_kwargs)

    dset_kwargs['h5_path'] = args.val_h5
    dset_kwargs['box_h5'] = args.val_box_h5
    if args.box_dataset:
        dset_kwargs['filter_curr_graphs'] = args.filter_curr_graphs
        val_dset = SceneGraphBoxDataset(**dset_kwargs)
    else:
        val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

    dset_kwargs['h5_path'] = args.test_h5
    dset_kwargs['box_h5'] = args.test_box_h5
    if args.box_dataset:
        dset_kwargs['filter_curr_graphs'] = args.filter_curr_graphs
        test_dset = SceneGraphBoxDataset(**dset_kwargs)
    else:
        test_dset = SceneGraphNoPairsDataset(**dset_kwargs)

    unchosen_train_dset = None
    class_ix = None

    print('\nObtained data in time: ', (time.time() - start))
    print('Len Train Dset ', len(train_dset))
    print('Len Val Dset ', len(val_dset))
    print('Len Test Dset ', len(test_dset))

    train_dset = RandomSplitReindex(train_dset)
    unchosen_train_dset = RandomSplitReindex(unchosen_train_dset)
    val_dset = RandomSplitReindex(val_dset)
    test_dset = RandomSplitReindex(test_dset)

    return vocab, train_dset, unchosen_train_dset, val_dset, test_dset, class_ix


def build_train_loaders(args, shuffle=True, return_class_ix=False, drop_last=False):
    print(args.dataset)
    vocab, train_dset, unchosen_train_dset, val_dset, test_dset, class_ix = build_train_dsets(args)
    if args.box_dataset:
        collate_fn = collate_fn_boxes
    else:
        collate_fn = collate_fn_nopairs

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    unchosen_train_loader = DataLoader(unchosen_train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    loader_kwargs['drop_last'] = False
    val_loader = DataLoader(val_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    test_loader = DataLoader(test_dset, **loader_kwargs)

    if return_class_ix:
        return vocab, train_loader, unchosen_train_loader, val_loader, test_loader, class_ix
    else:
        return vocab, train_loader, unchosen_train_loader, val_loader, test_loader


def make_qtype_h5_file(data_loader, h5_file_full_path, data_type):
    if os.path.exists(h5_file_full_path):
        # os.remove(h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    h5_file = h5py.File(h5_file_full_path, 'w')
    if data_type == 'train':
        dset = h5_file.create_dataset("train", (3474969, 5))
    elif data_type == 'val':
        dset = h5_file.create_dataset("val", (279273, 5))
    elif data_type == 'test':
        dset = h5_file.create_dataset("test", (281739, 5))
    qtype_counter = 0
    with torch.no_grad():
        for batch_ix, target in tqdm(enumerate(data_loader), total=len(data_loader)):

            # grab all data for current subject main box
            # orig_box_id = target['box_id']
            img_id = int(target['img_id'])
            # image = target['image']
            # main_box_id = target['main_box_id']
            # related_box_ids = target['related_box_ids']
            # obj_labels = target['obj_labels']
            # boxes = target['boxes']
            predicates = target['predicates']
            attributes = target['attributes']
            # box_features = target['box_features']

            questions = []
            if len(predicates) > 0:
                for (subj, pred, obj) in predicates:
                    subj = int(subj.item())
                    pred = int(pred.item())
                    obj = int(obj.item())
                    question1 = [img_id, int(QType.spos), subj, pred, obj]
                    question2 = [img_id, int(QType.spop), subj, pred, obj]
                    question3 = [img_id, int(QType.spoo), subj, pred, obj]
                    questions.append(question1)
                    questions.append(question2)
                    questions.append(question3)

            has_attr_pred = 0
            for (subj, attr) in attributes:
                subj = int(subj.item())
                attr = int(attr.item())
                question1 = [img_id, int(QType.spas), subj, has_attr_pred, attr]
                question2 = [img_id, int(QType.spap), subj, has_attr_pred, attr]
                question3 = [img_id, int(QType.spaa), subj, has_attr_pred, attr]
                questions.append(question1)
                questions.append(question2)
                questions.append(question3)

            # save out data for each question
            for ques in questions:
                dset[qtype_counter] = np.array(ques)
                qtype_counter += 1
    h5_file.close()


def make_scene_graph_h5(data_loader, triple_h5_file_full_path, attribute_h5_file_full_path):
    if os.path.exists(triple_h5_file_full_path):
        # os.remove(triple_h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    if os.path.exists(attribute_h5_file_full_path):
        # os.remove(attribute_h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    triple_h5_file = h5py.File(triple_h5_file_full_path, 'w')
    attribute_h5_file = h5py.File(attribute_h5_file_full_path, 'w')
    visited_img_id = []
    with torch.no_grad():
        for batch_ix, target in tqdm(enumerate(data_loader), total=len(data_loader)):
            triples = target['predicates']
            attributes = target['attributes']
            img_id = target['img_id'].item()
            if img_id in visited_img_id:
                continue
            visited_img_id.append(img_id)
            triple_h5_file.create_dataset(str(img_id), data=triples.cpu().numpy())
            attribute_h5_file.create_dataset(str(img_id), data=attributes.cpu().numpy())
    triple_h5_file.close()
    attribute_h5_file.close()


def main_qtype(args):
    # load Visual Genome data
    vocab, train_loader, _, val_loader, test_loader = build_train_loaders(args, return_class_ix=False, drop_last=False,
                                                                          shuffle=False)

    # save out question type h5 files
    make_qtype_h5_file(val_loader, os.path.join(args.output_dir, "vg_qtype_dset_val.h5"), data_type='val')
    make_qtype_h5_file(test_loader, os.path.join(args.output_dir, "vg_qtype_dset_test.h5"), data_type='test')
    make_qtype_h5_file(train_loader, os.path.join(args.output_dir, "vg_qtype_dset_train.h5"), data_type='train')


def main_scene_graphs(args):
    # load Visual Genome data
    vocab, train_loader, _, val_loader, test_loader = build_train_loaders(args, shuffle=False, return_class_ix=False,
                                                                          drop_last=False)

    # save box coordinates to h5 files
    make_scene_graph_h5(train_loader, os.path.join(args.output_dir, "vg_scene_graphs_triples1_train.h5"),
                        os.path.join(args.output_dir, "vg_scene_graphs_attributes1_train.h5"))

    make_scene_graph_h5(val_loader, os.path.join(args.output_dir, "vg_scene_graphs_triples1_val.h5"),
                        os.path.join(args.output_dir, "vg_scene_graphs_attributes1_val.h5"))

    make_scene_graph_h5(test_loader, os.path.join(args.output_dir, "vg_scene_graphs_triples1_test.h5"),
                        os.path.join(args.output_dir, "vg_scene_graphs_attributes1_test.h5"))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.batch_size = 1
    args.shuffle_val = False
    args.num_train_samples = -1
    args.output_dir = META_DATA_DIR
    args.filter_curr_graphs = True
    main_qtype(args)
    args.filter_curr_graphs = False
    main_scene_graphs(args)
