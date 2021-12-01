#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np
import h5py
import PIL


def get_left_right_top_bottom(box, height, width):
    """
    - box: Tensor of size [4]
    - height: scalar, image hight
    - width: scalar, image width
    return: left, right, top, bottom in image coordinates
    """
    left = (box[0] * width).type(torch.int32)
    right = (box[2] * width).type(torch.int32)
    top = (box[1] * height).type(torch.int32)
    bottom = (box[3] * height).type(torch.int32)

    return left, right, top, bottom


class SceneGraphNoPairsDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, clean_repeats=True, predgraphs=False, use_unnormalized_boxes=False,
                 box_h5=None, include_dummy_items=True, num_attrs=75):
        super(SceneGraphNoPairsDataset, self).__init__()

        self.image_dir = image_dir
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_unnormalized_boxes = use_unnormalized_boxes
        self.box_h5 = box_h5
        self.predgraphs = predgraphs
        self.include_dummy_items = include_dummy_items
        self.num_attrs = num_attrs

        self.clean_repeats = clean_repeats

        # standard object detection just uses totensor
        self.transform = T.ToTensor()

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

        if box_h5 is not None:
            self.box_data = {}
            with h5py.File(box_h5, 'r') as f:
                for k, v in f.items():
                    self.box_data[k] = torch.from_numpy(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        # img_path = os.path.join(self.image_dir, self.image_paths[index])
        img_path = os.path.join(self.image_dir, self.image_paths[index].split('/')[-1])  # all images in one folder
        img_id = int(self.image_paths[index].split('/')[-1].split('.')[0])  # integer image id

        if self.box_h5 is not None:
            box_features = self.box_data[str(img_id)]
        else:
            box_features = torch.empty((0, 1024))  # dummy placeholder features

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                # print(WW, HH)
                image = self.transform(image.convert('RGB'))

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        obj_idxs += obj_idxs_without_rels  # add all objects without relationships
        map_overlapping_obj = {}

        objs = []
        boxes = []
        boxes_unnorm = []

        obj_idx_mapping = {}
        counter = 0
        for i, obj_idx in enumerate(obj_idxs):

            curr_obj = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()

            x0 = float(x) / WW
            x0_unnorm = float(x)
            y0 = float(y) / HH
            y0_unnorm = float(y)
            if self.predgraphs:
                x1 = float(w) / WW
                x1_unnorm = float(w)
                y1 = float(h) / HH
                y1_unnorm = float(h)
            else:
                x1 = float(x + w) / WW
                x1_unnorm = float(x + w)
                y1 = float(y + h) / HH
                y1_unnorm = float(y + h)

            curr_box = torch.FloatTensor([x0, y0, x1, y1])
            curr_box_unnrom = torch.LongTensor([x0_unnorm, y0_unnorm, x1_unnorm, y1_unnorm])

            found_overlap = False
            if self.predgraphs:
                for prev_idx in range(counter):
                    if overlapping_nodes(objs[prev_idx], curr_obj, boxes[prev_idx], curr_box):
                        map_overlapping_obj[i] = prev_idx
                        found_overlap = True
                        break
            if not found_overlap:
                objs.append(curr_obj)
                boxes.append(curr_box)
                boxes_unnorm.append(curr_box_unnrom)
                map_overlapping_obj[i] = counter
                counter += 1

            obj_idx_mapping[obj_idx] = map_overlapping_obj[i]

        # The last object will be the special __image__ object
        objs.append(self.vocab['object_name_to_idx']['__image__'])
        boxes.append(torch.FloatTensor([0, 0, 1, 1]))
        boxes_unnorm.append(torch.FloatTensor([0, 0, WW, HH]))

        boxes = torch.stack(boxes)
        boxes_unnorm = torch.stack(boxes_unnorm)
        objs = torch.LongTensor(objs)
        num_objs = counter + 1

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                if self.clean_repeats and [s, p, o] in triples:
                    continue
                if self.predgraphs and s == o:
                    continue
                triples.append([s, p, o])

        if self.include_dummy_items:
            # Add dummy __in_image__ relationships for all objects
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(num_objs - 1):
                triples.append([i, in_image, num_objs - 1])

        triples = torch.LongTensor(triples)

        attributes = []
        attribute_idxs = self.data['attribute_idxs'][index]
        for iii in obj_idxs:
            o = obj_idx_mapping.get(iii, None)
            if o is None:
                continue

            attribute_idx = attribute_idxs[iii].item()
            o_attr = self.data['object_attributes'][attribute_idx]
            for ii in range(self.data['attributes_per_object'][index][iii].item()):
                attributes.append([o, o_attr[ii]])

            # append object label as attribute too
            obj_label = objs[
                            iii] + self.num_attrs - 1  # shift by attribute labels and subtract 1 due to dummy object category at 0 index
            attributes.append([o, obj_label])

        if self.include_dummy_items:
            # Add dummy __attr__ attributes for all objects
            dummy_attr = self.vocab['attribute_name_to_idx']['__attr__']
            for i in range(num_objs - 1):
                attributes.append([i, dummy_attr])

        attributes = torch.LongTensor(attributes)

        if self.use_unnormalized_boxes:
            boxes_r = boxes_unnorm
        else:
            boxes_r = boxes
        return img_id, image, objs, boxes_r, triples, attributes, box_features


def collate_fn_nopairs(batch):
    """
    Collate function to be used when wrapping a SceneGraphNoPairsDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, 3, H, W)
    - objs: LongTensor of shape (num_objs,) giving categories for all objects
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
    - triples: FloatTensor of shape (num_triples, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n]
    - imgs_masked: FloatTensor of shape (N, 4, H, W)
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_img_id, all_imgs, all_objs, all_boxes, all_triples, all_attributes, all_box_features = [], [], [], [], [], [], []
    all_obj_to_img, all_triple_to_img, all_attribute_to_img = [], [], []

    obj_offset = 0

    for i, (img_id, img, objs, boxes, triples, attributes, box_features) in enumerate(batch):
        all_imgs.append(img[None])
        all_img_id.append(img_id)
        num_objs, num_triples, num_attributes = objs.size(0), triples.size(0), attributes.size(0)

        all_objs.append(objs)
        all_boxes.append(boxes)
        all_box_features.append(box_features)
        triples = triples.clone()
        attributes = attributes.clone()

        if len(triples) > 0:
            triples[:, 0] += obj_offset
            triples[:, 2] += obj_offset
        attributes[:, 0] += obj_offset  # always have at least object label as attribute

        all_triples.append(triples)
        all_attributes.append(attributes)

        all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
        all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))
        all_attribute_to_img.append(torch.LongTensor(num_attributes).fill_(i))

        obj_offset += num_objs

    all_img_id = torch.LongTensor(all_img_id)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_box_features = torch.cat(all_box_features)
    all_triples = torch.cat(all_triples)
    all_attributes = torch.cat(all_attributes)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_attribute_to_img = torch.cat(all_attribute_to_img)

    return all_img_id, all_imgs, all_objs, all_boxes, all_triples, all_attributes, all_box_features, \
           all_obj_to_img, all_triple_to_img, all_attribute_to_img


def overlapping_nodes(obj1, obj2, box1, box2, criteria=0.7):
    # used to clean predicted graphs - merge nodes with overlapping boxes
    # are these two objects overplapping?
    # boxes given as [left, top, right, bottom]
    res = 100  # used to project box representation in 2D for iou computation
    epsilon = 0.001
    if obj1 == obj2:
        spatial_box1 = np.zeros([res, res])
        left, right, top, bottom = get_left_right_top_bottom(box1, res, res)
        spatial_box1[top:bottom, left:right] = 1
        spatial_box2 = np.zeros([res, res])
        left, right, top, bottom = get_left_right_top_bottom(box2, res, res)
        spatial_box2[top:bottom, left:right] = 1
        iou = np.sum(spatial_box1 * spatial_box2) / \
              (np.sum((spatial_box1 + spatial_box2 > 0).astype(np.float32)) + epsilon)
        return iou >= criteria
    else:
        return False
