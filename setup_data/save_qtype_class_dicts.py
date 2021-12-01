import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict

from utils import QType
from train_continual_active_learners.vg_qtype_dset_loader_utils import build_loaders
from train_continual_active_learners.train_triple_completion_qtype import get_args_parser
from config import META_DATA_DIR


def make_dicts(data_loader, pred_save_file, attr_save_file):
    attr_dict = defaultdict(list)
    pred_dict = defaultdict(list)

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        index = batch['index']
        question_id = batch['question_id']
        predicate_labels = batch['predicates']
        attr_labels = batch['attributes']

        # determine which questions contain attributes
        attr_q = torch.cat([torch.where(question_id == QType.spap)[0], torch.where(question_id == QType.spas)[0],
                            torch.where(question_id == QType.spaa)[0]])

        # store predicates in dictionary
        for pred, id in zip(predicate_labels, index):
            pred_dict[pred.item()].append(id.item())

        # store attributes in dictionary
        for attr, id in zip(attr_labels[attr_q], index[attr_q]):
            attr_dict[attr.item()].append(id.item())

    # save dictionaries out to json files
    print('\nSaving predicate dictionary out to: %s' % pred_save_file)
    with open(pred_save_file, 'w', encoding="utf8") as f_pred:
        json.dump(pred_dict, f_pred)

    print('\nSaving attribute dictionary out to: %s' % attr_save_file)
    with open(attr_save_file, 'w', encoding="utf8") as f_attr:
        json.dump(attr_dict, f_attr)


def main(args):
    vocab, train_loader, _, _, _, val_loader, test_loader, _ = build_loaders(args, drop_last=False, shuffle=False)
    pred_base_file = os.path.join(args.output_dir, 'vg_qtype_predicate_dict_%s.json')
    attr_base_file = os.path.join(args.output_dir, 'vg_qtype_attribute_dict_%s.json')

    print('\nWorking on val data...')
    data_type = 'val'
    make_dicts(val_loader, pred_base_file % data_type, attr_base_file % data_type)

    print('\nWorking on test data...')
    data_type = 'test'
    make_dicts(test_loader, pred_base_file % data_type, attr_base_file % data_type)

    print('\nWorking on train data...')
    data_type = 'train'
    make_dicts(train_loader, pred_base_file % data_type, attr_base_file % data_type)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.shuffle_val = False
    args.num_train_samples = -1
    args.output_dir = META_DATA_DIR
    main(args)
