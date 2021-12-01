import os
import PIL

import torch

from vg_image_dset import SceneGraphNoPairsDataset, overlapping_nodes


class SceneGraphBoxDataset(SceneGraphNoPairsDataset):
    def __init__(self, vocab, h5_path, image_dir, clean_repeats=True, predgraphs=False, use_unnormalized_boxes=False,
                 box_h5=None, num_object_categories=178, num_attribute_categories=75, num_relationship_categories=45,
                 num_attrs=75, filter_curr_graphs=False):
        assert box_h5 is not None
        super(SceneGraphBoxDataset, self).__init__(vocab, h5_path, image_dir, clean_repeats=clean_repeats,
                                                   predgraphs=predgraphs, use_unnormalized_boxes=use_unnormalized_boxes,
                                                   box_h5=box_h5)

        self.num_object_categories = num_object_categories
        self.num_attribute_categories = num_attribute_categories
        self.num_relationship_categories = num_relationship_categories
        self.filter_curr_graphs = filter_curr_graphs

        # convert image paths list to dictionary with keys as image ids
        self.image_paths_dict = {}
        for path in self.image_paths:
            img_id = path.split('/')[1].split('.')[0]
            self.image_paths_dict[img_id] = path

        # partition images out into boxes for convenience
        self.box_id_dict = {}
        count = 0
        for k in list(self.image_paths_dict.keys()):
            img_id = k
            box_feats = self.box_data[k]
            for box_id, feat in enumerate(box_feats[:-1]):  # drop full image box feature
                self.box_id_dict[count] = [img_id, box_id]
                count += 1

        self.num_attrs = num_attrs

    def __len__(self):
        return len(self.box_id_dict)

    def __getitem__(self, box_id):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        box_info = self.box_id_dict[box_id]  # get meta data for local box index
        img_id, relative_img_box_id = box_info
        img_path = self.image_paths_dict[img_id].split('/')[-1]  # remove data dir since all images in same dir
        img_path = os.path.join(self.image_dir, img_path)
        index = list(self.image_paths_dict.keys()).index(img_id)  # get local index in self.data array

        box_features = self.box_data[img_id][:-1]  # drop full image box feature at end of array
        img_id = int(img_id)

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
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
            x0_unnorm = x
            y0 = float(y) / HH
            y0_unnorm = y

            x1 = float(x + w) / WW
            x1_unnorm = x + w
            y1 = float(y + h) / HH
            y1_unnorm = y + h

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
                triples.append([s, p, o])

        triples = torch.LongTensor(triples)

        attributes = []
        attribute_idxs = self.data['attribute_idxs'][index]
        for j, iii in enumerate(obj_idxs):
            o = obj_idx_mapping.get(iii, None)
            if o is None:
                continue

            attribute_idx = attribute_idxs[iii].item()
            o_attr = self.data['object_attributes'][attribute_idx]
            for ii in range(self.data['attributes_per_object'][index][iii].item()):
                attributes.append([o, o_attr[ii]])

            if not self.filter_curr_graphs:
                # append object label as attribute too
                obj_label = objs[
                                j] + self.num_attrs - 1  # shift by attribute labels and subtract 1 due to dummy object category at 0 index
                attributes.append([o, obj_label])

        attributes = torch.LongTensor(attributes)

        if self.use_unnormalized_boxes:
            boxes_r = boxes_unnorm
        else:
            boxes_r = boxes

        if self.filter_curr_graphs:
            # only grab triples associated with current box
            bool_subject = triples[:, 0] == relative_img_box_id
            triples_curr_box = triples[bool_subject]

            related_obj_ids = torch.cat(
                [torch.tensor(relative_img_box_id).unsqueeze(0), torch.unique(triples_curr_box[:, 2])])

            # add object category as attribute
            obj_list = torch.LongTensor(
                [relative_img_box_id, objs[relative_img_box_id] + self.num_attrs - 1]).unsqueeze(
                0)  # shift by attribute labels and subtract 1 due to dummy object category at 0 index
            attributes = torch.cat([attributes, obj_list], dim=0)

            # only grab attributes associated with current box
            bool_attr = attributes[:, 0] == relative_img_box_id
            attributes_curr_box = attributes[bool_attr]
        else:
            triples_curr_box = triples
            attributes_curr_box = attributes
            related_obj_ids = torch.unique(
                torch.cat([torch.unique(triples_curr_box[:, 2]), torch.unique(triples_curr_box[:, 0])]))

        # make dictionary of targets
        target = {}
        target['box_id'] = box_id
        target['img_id'] = img_id
        target['image'] = image
        target['main_box_id'] = relative_img_box_id
        target['related_box_ids'] = related_obj_ids
        target['obj_labels'] = objs
        target['boxes'] = boxes_r
        target['predicates'] = triples_curr_box
        target['attributes'] = attributes_curr_box
        target['box_features'] = box_features

        return target


def collate_fn_boxes(batch):
    """
    Collate function to be used when wrapping a SceneGraphBoxDataset in a
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
    all_orig_box_ids, all_img_id, all_imgs, all_main_box_ids, all_rel_box_ids, all_obj_labels, all_boxes, all_predicates, all_attributes, all_box_features = [], [], [], [], [], [], [], [], [], []
    all_box_feature_id_to_main_box_id = []

    obj_offset = 0

    for i, target in enumerate(batch):

        orig_box_id = target['box_id']
        img_id = target['img_id']
        img = target['image']
        main_box_id = target['main_box_id']
        related_box_ids = target['related_box_ids']
        labels = target['obj_labels']
        boxes = target['boxes']
        predicates = target['predicates']
        attributes = target['attributes']
        box_features = target['box_features']

        all_imgs.append(img[None])
        all_img_id.append(img_id)
        all_orig_box_ids.append(orig_box_id)
        num_objs, num_triples, num_attributes = labels.size(0), predicates.size(0), attributes.size(0)

        all_obj_labels.append(labels)
        all_boxes.append(boxes)
        all_box_features.append(box_features)

        predicates = predicates.clone()
        attributes = attributes.clone()
        if len(predicates) > 0:
            predicates[:, 0] += obj_offset
            predicates[:, 2] += obj_offset
        attributes[:, 0] += obj_offset
        all_main_box_ids.append(main_box_id + obj_offset)
        all_rel_box_ids.append(related_box_ids + obj_offset)

        all_predicates.append(predicates)
        all_attributes.append(attributes)

        all_box_feature_id_to_main_box_id.append(torch.LongTensor(len(box_features)).fill_(main_box_id + obj_offset))

        obj_offset += num_objs

    all_orig_box_ids = torch.LongTensor(all_orig_box_ids)
    all_img_id = torch.LongTensor(all_img_id)
    all_main_box_ids = torch.LongTensor(all_main_box_ids)
    all_rel_box_ids = torch.cat(all_rel_box_ids)
    all_obj_labels = torch.cat(all_obj_labels)
    all_boxes = torch.cat(all_boxes)
    all_predicates = torch.cat(all_predicates)
    all_attributes = torch.cat(all_attributes)
    all_box_features = torch.cat(all_box_features)
    all_box_feature_id_to_main_box_id = torch.cat(all_box_feature_id_to_main_box_id)

    # make dictionary of targets
    target = {}
    target['box_id'] = all_orig_box_ids
    target['img_id'] = all_img_id
    target['image'] = all_imgs
    target['main_box_id'] = all_main_box_ids
    target['related_box_ids'] = all_rel_box_ids
    target['obj_labels'] = all_obj_labels
    target['boxes'] = all_boxes
    target['predicates'] = all_predicates
    target['attributes'] = all_attributes
    target['box_features'] = all_box_features
    target['box_feature_ids_to_main_box_ids'] = all_box_feature_id_to_main_box_id

    return target
