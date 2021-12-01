import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL
from enum import IntEnum


class QType(IntEnum):
    '''
    Format: (subject, predicate, object/attribute, missing element)
    '''
    spos = 0
    spas = 1
    spop = 2
    spap = 3
    spoo = 4
    spaa = 5


class SceneGraphQuestionDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, dset_type, box_feature_h5=None, box_h5=None, question_h5=None,
                 triple_h5=None, attribute_h5=None, load_img=False):
        super(SceneGraphQuestionDataset, self).__init__()

        self.image_dir = image_dir
        self.vocab = vocab
        self.load_img = load_img

        if load_img:
            # original vg image data h5
            self.data = {}
            with h5py.File(h5_path, 'r') as f:
                for k, v in f.items():
                    if k == 'image_paths':
                        self.image_paths = list(v)

            # convert image paths list to dictionary with keys as image ids
            self.image_paths_dict = {}
            for path in self.image_paths:
                img_id = path.split('/')[1].split('.')[0]
                self.image_paths_dict[img_id] = path

            self.transform = T.ToTensor()

        # h5 with bounding box features
        self.mega_data = {}
        with h5py.File(box_feature_h5, 'r') as f:
            for k, v in f.items():
                self.mega_data[int(k)] = [torch.from_numpy(np.asarray(v))]

        # h5 with unnormalized bounding box coordinates
        with h5py.File(box_h5, 'r') as f:
            for k, v in f.items():
                self.mega_data[int(k)].append(torch.from_numpy(np.asarray(v)))

        # h5 with scene graph triples
        with h5py.File(triple_h5, 'r') as f:
            for k, v in f.items():
                self.mega_data[int(k)].append(torch.from_numpy(np.asarray(v)))

        # h5 with scene graph attributes
        with h5py.File(attribute_h5, 'r') as f:
            for k, v in f.items():
                self.mega_data[int(k)].append(torch.from_numpy(np.asarray(v)))

        # h5 with question data
        with h5py.File(question_h5, 'r') as f:
            self.question_data = torch.from_numpy(np.asarray(f[dset_type])).long()

    def __len__(self):
        return self.question_data.shape[0]

    def __getitem__(self, index):

        # grab current question data
        curr_question_data = self.question_data[index]
        img_id = int(curr_question_data[0].item())
        question_id = curr_question_data[1]
        subject_box_id = curr_question_data[2]
        predicate = curr_question_data[3]

        # get box and graph
        box_features, boxes, triples_graph, attributes_graph = self.mega_data[img_id]

        # grab appropriate data based on question type
        if question_id in [QType.spos, QType.spop, QType.spoo]:
            # object question
            object_box_id = curr_question_data[-1]
            attribute = 0  # dummy value
        else:
            # attribute question
            attribute = curr_question_data[-1]
            object_box_id = -1000000000  # dummy value

        # load image if desired
        if self.load_img:
            img_path = self.image_paths_dict[img_id]
            img_path = os.path.join(self.image_dir, img_path)
            with open(img_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    # WW, HH = image.size
                    image = self.transform(image.convert('RGB'))
        else:
            image = None

        # make dictionary of targets
        target = {}
        target['question_id'] = question_id
        target['img_id'] = img_id
        target['image'] = image
        target['boxes'] = boxes
        target['box_features'] = box_features
        target['subject_box_id'] = subject_box_id
        target['object_box_id'] = object_box_id
        target['predicate'] = predicate
        target['attribute'] = attribute
        target['original_index'] = index
        target['index'] = index
        target['triples_graph'] = triples_graph
        target['attributes_graph'] = attributes_graph

        return target


def collate_fn_qtype(batch):
    """
    Collate function to be used when wrapping a SceneGraphQuestionDataset in a
    DataLoader
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_indices, all_orig_indices, all_question_id, all_img_id, all_imgs, all_boxes, all_box_features, all_subject_box_id, all_object_box_id, all_predicates, all_attributes = [], [], [], [], [], [], [], [], [], [], []
    all_box_feature_id_to_subject_box_id = []
    all_triples_graph = []
    all_attributes_graph = []
    all_relative_subject_box_id = []
    all_relative_object_box_id = []

    obj_offset = 0

    for i, target in enumerate(batch):

        index = target['index']
        original_index = target['original_index']
        question_id = target['question_id']
        img_id = target['img_id']
        img = target['image']
        boxes = target['boxes']
        box_features = target['box_features']
        subject_box_id = target['subject_box_id']
        object_box_id = target['object_box_id']
        predicate = target['predicate']
        attribute = target['attribute']
        triples_graph = target['triples_graph']
        attributes_graph = target['attributes_graph']

        if img is not None:
            all_imgs.append(img[None])
        all_img_id.append(img_id)
        all_question_id.append(question_id)
        all_indices.append(index)
        all_orig_indices.append(original_index)

        all_boxes.append(boxes)
        all_box_features.append(box_features)
        num_objs = boxes.shape[0]

        all_subject_box_id.append(subject_box_id + obj_offset)
        all_object_box_id.append(object_box_id + obj_offset)
        all_relative_subject_box_id.append(subject_box_id)
        all_relative_object_box_id.append(object_box_id)
        all_triples_graph.append(triples_graph)
        all_attributes_graph.append(attributes_graph)

        all_predicates.append(predicate)
        all_attributes.append(attribute)

        all_box_feature_id_to_subject_box_id.append(torch.LongTensor(num_objs).fill_(subject_box_id + obj_offset))

        obj_offset += num_objs

    all_indices = torch.LongTensor(all_indices)
    all_orig_indices = torch.LongTensor(all_orig_indices)
    all_question_id = torch.LongTensor(all_question_id)
    all_img_id = torch.LongTensor(all_img_id)
    # all_imgs = torch.cat(all_imgs) # images are different shapes and must be returned in a list
    all_boxes = torch.cat(all_boxes)
    all_box_features = torch.cat(all_box_features)
    all_subject_box_id = torch.LongTensor(all_subject_box_id)
    all_object_box_id = torch.LongTensor(all_object_box_id)
    all_relative_subject_box_id = torch.LongTensor(all_relative_subject_box_id)
    all_relative_object_box_id = torch.LongTensor(all_relative_object_box_id)
    all_predicates = torch.LongTensor(all_predicates)
    all_attributes = torch.LongTensor(all_attributes)
    all_box_feature_id_to_subject_box_id = torch.cat(all_box_feature_id_to_subject_box_id)

    # make dictionary of targets
    target = {}
    target['index'] = all_indices
    target['original_index'] = all_orig_indices
    target['question_id'] = all_question_id
    target['img_id'] = all_img_id
    target['image'] = all_imgs
    target['boxes'] = all_boxes
    target['box_features'] = all_box_features
    target['subject_box_id'] = all_subject_box_id
    target['object_box_id'] = all_object_box_id
    target['relative_subject_box_id'] = all_relative_subject_box_id
    target['relative_object_box_id'] = all_relative_object_box_id
    target['predicates'] = all_predicates
    target['attributes'] = all_attributes
    target['triples_graph'] = all_triples_graph
    target['attributes_graph'] = all_attributes_graph

    target['box_feature_ids_to_subject_box_ids'] = all_box_feature_id_to_subject_box_id

    return target
