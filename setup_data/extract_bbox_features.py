import argparse, os
import h5py
from tqdm import tqdm
import json

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

from train_object_detection_feature_extractor.rcnn_components import get_features, FasterRCNNModified
from vg_image_dset import SceneGraphNoPairsDataset, collate_fn_nopairs
from utils import bool_flag
from config import DATA_DIR, META_DATA_DIR, COCO_CKPT


def parse_args(add_help=False):
    # helps parsing the same arguments in a different script
    parser = argparse.ArgumentParser(description='Load VG Dataset', add_help=add_help)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--batch_size', default=32, type=int)

    # Dataset options
    parser.add_argument('--num_train_samples', type=int, default=-1)
    parser.add_argument('--shuffle_val', default=True, type=bool_flag)
    parser.add_argument('--loader_num_workers', default=8, type=int)
    parser.add_argument('--vg_image_dir', default=os.path.join(DATA_DIR, ''))
    parser.add_argument('--train_h5', default=os.path.join(META_DATA_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(META_DATA_DIR, 'val.h5'))
    parser.add_argument('--test_h5', default=os.path.join(META_DATA_DIR, 'test.h5'))
    parser.add_argument('--vocab_json', default=os.path.join(META_DATA_DIR, 'vocab.json'))
    parser.add_argument('--use_unnormalized_boxes', default=True, type=bool_flag)

    # bounding box feature h5 files
    parser.add_argument('--train_box_h5', default=None)  # os.path.join(META_DATA_DIR, 'vg_box_rn50_features_train.h5'))
    parser.add_argument('--val_box_h5', default=None)  # os.path.join(META_DATA_DIR, 'vg_box_rn50_features_val.h5'))
    parser.add_argument('--test_box_h5', default=None)  # os.path.join(META_DATA_DIR, 'vg_box_rn50_features_test.h5'))

    parser.add_argument('--output_dir', type=str, default=META_DATA_DIR)
    parser.add_argument('--coco_pretrain_ckpt', type=str, default=COCO_CKPT)
    parser.add_argument('--include_dummy_items', type=bool_flag,
                        default=False)  # exclude 'in_image' and 'has_attr' predicates and attributes
    args = parser.parse_args()
    args.batch_size = 1
    args.shuffle_val = False
    print(args)
    return args


def build_train_dsets(args):
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
        'include_dummy_items': args.include_dummy_items,
    }
    train_dset = SceneGraphNoPairsDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per train epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    dset_kwargs['box_h5'] = args.val_box_h5
    val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

    dset_kwargs['h5_path'] = args.test_h5
    dset_kwargs['box_h5'] = args.test_box_h5
    test_dset = SceneGraphNoPairsDataset(**dset_kwargs)

    return vocab, train_dset, val_dset, test_dset


def build_train_loaders(args, shuffle=True):
    print(args.dataset)
    vocab, train_dset, val_dset, test_dset = build_train_dsets(args)
    collate_fn = collate_fn_nopairs

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = False
    test_loader = DataLoader(test_dset, **loader_kwargs)

    return vocab, train_loader, val_loader, test_loader


def get_model(num_classes, ckpt_file=None):
    # load a pre-trained model for classification and return
    # only the features
    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(res50_model.children())[:-2])
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 2048

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNNModified(backbone,
                               num_classes=num_classes,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler)

    if ckpt_file is not None:
        print('\nLoading model from: %s' % ckpt_file)
        state_dict = torch.load(ckpt_file)['model']
        model.load_state_dict(state_dict)
    return model


def extract_box_features(model, data_loader, h5_file_full_path, device):
    if os.path.exists(h5_file_full_path):
        # os.remove(h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    h5_file = h5py.File(h5_file_full_path, 'w')
    with torch.no_grad():
        for batch_ix, (image_id, image, labels, boxes, _, _, _, _, _, _) in tqdm(enumerate(data_loader),
                                                                                 total=len(data_loader)):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            num_objs = len(boxes)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            target = {}
            target["boxes"] = boxes.long().to(device)
            target["labels"] = labels.to(device)
            target["image_id"] = image_id.to(device)
            target["area"] = area.to(device)
            target["iscrowd"] = iscrowd.to(device)
            image = image[0].to(device).squeeze()

            # extract box features
            box_feats_rpn, labels_rpn, _, box_feats_gt, labels_gt = get_features(model, [image], [target])
            # print(image_id, box_feats_gt.shape)
            h5_file.create_dataset(str(image_id.item()), data=box_feats_gt.cpu().numpy())
    h5_file.close()


def save_box_coordinates(data_loader, h5_file_full_path):
    if os.path.exists(h5_file_full_path):
        # os.remove(h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    h5_file = h5py.File(h5_file_full_path, 'w')
    with torch.no_grad():
        for batch_ix, (image_id, image, labels, boxes, _, _, _, _, _, _) in tqdm(enumerate(data_loader),
                                                                                 total=len(data_loader)):
            boxes = boxes.long()
            h5_file.create_dataset(str(image_id.item()), data=boxes.cpu().numpy())
    h5_file.close()


def main_box_features(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load a Faster RCNN model pre-trained on COCO
    model = get_model(num_classes=91, ckpt_file=args.coco_pretrain_ckpt)
    model.eval()
    model.to(device)

    # load Visual Genome data
    vocab, train_loader, val_loader, test_loader = build_train_loaders(args, shuffle=False)

    # extract box features and save to h5 files
    extract_box_features(model, train_loader, os.path.join(args.output_dir, "vg_box_rn50_features_train.h5"),
                         device=device)

    extract_box_features(model, val_loader, os.path.join(args.output_dir, "vg_box_rn50_features_val.h5"), device=device)

    extract_box_features(model, test_loader, os.path.join(args.output_dir, "vg_box_rn50_features_test.h5"),
                         device=device)


def main_box_coordinates(args):
    # load Visual Genome data
    vocab, train_loader, val_loader, test_loader = build_train_loaders(args, shuffle=False)

    # save box coordinates to h5 files
    save_box_coordinates(train_loader, os.path.join(args.output_dir, "vg_box_coordinates_train.h5"))

    save_box_coordinates(val_loader, os.path.join(args.output_dir, "vg_box_coordinates_val.h5"))

    save_box_coordinates(test_loader, os.path.join(args.output_dir, "vg_box_coordinates_test.h5"))


if __name__ == '__main__':
    args = parse_args()
    main_box_features(args)
    main_box_coordinates(args)
