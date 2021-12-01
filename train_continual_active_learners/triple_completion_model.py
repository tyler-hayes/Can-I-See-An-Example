import torch
from torch import nn
import numpy as np
from mish import Mish
import torch.nn.functional as F

from utils import QType


class CompletionModel(nn.Module):
    def __init__(self, bbox_encoder_shape, prediction_network_shape, num_predicates, num_attributes,
                 normalize_embeddings=False, tied_target_weights=True, learnable_null=True, dropout_prob=0.,
                 final_activation='sigmoid', normalization='bn'):
        super(CompletionModel, self).__init__()

        # define variables
        self.embedding_dimension = prediction_network_shape[-1]
        self.normalize_embeddings = normalize_embeddings
        self.tied_target_weights = tied_target_weights
        self.dropout_prob = dropout_prob  # probability of an element being zeroed

        if normalization == 'bn':
            self.norm = nn.BatchNorm1d
        elif normalization == 'ln':
            self.norm = nn.LayerNorm
        else:
            raise NotImplementedError

        # make embedding network for each type of data (bbox, attribute, predicate)
        self.bbox_encoder, self.bbox_encoders_ref = self.make_encoding_network(bbox_encoder_shape, name="bbox_encoder_",
                                                                               final_activation=final_activation)
        self.pred_encoder = nn.Embedding(num_predicates, self.embedding_dimension)
        self.attr_encoder = nn.Embedding(num_attributes, self.embedding_dimension)

        self.pre_train_parameters = [self.bbox_encoder.parameters()]
        self.non_pre_train_parameters = [self.pred_encoder.parameters(), self.attr_encoder.parameters()]

        if not tied_target_weights:
            # additional weights for target embeddings
            self.bbox_encoder_target, self.bbox_encoders_ref_target = self.make_encoding_network(bbox_encoder_shape,
                                                                                                 name="bbox_encoder_target_",
                                                                                                 final_activation=final_activation)
            self.pred_encoder_target = nn.Embedding(num_predicates, self.embedding_dimension)
            self.attr_encoder_target = nn.Embedding(num_attributes, self.embedding_dimension)
            self.pre_train_parameters.append(self.bbox_encoder_target.parameters())
            self.non_pre_train_parameters.append(self.pred_encoder_target.parameters())
            self.non_pre_train_parameters.append(self.attr_encoder_target.parameters())

        # make null vector for each type of data (bbox, attribute, predicate)
        self.box_null = torch.nn.Parameter(torch.randn(1, self.embedding_dimension), requires_grad=learnable_null)
        self.pred_null = torch.nn.Parameter(torch.randn(1, self.embedding_dimension), requires_grad=learnable_null)
        self.attr_null = torch.nn.Parameter(torch.randn(1, self.embedding_dimension), requires_grad=learnable_null)

        # make prediction network
        self.prediction_network, self.prediction_network_ref = self.make_encoding_network(prediction_network_shape,
                                                                                          name='prediction_network_',
                                                                                          final_activation=final_activation)
        self.pre_train_parameters.append(self.box_null)
        self.pre_train_parameters.append(self.prediction_network.parameters())
        self.non_pre_train_parameters.append(self.pred_null)
        self.non_pre_train_parameters.append(self.attr_null)

        # keep track of visited categories for label masking
        self.visited_attribute_categories = []
        self.visited_predicate_categories = []

        # initialize dictionaries for counts of seen attributes, predicates, and qtypes
        self.visited_attribute_counts = {}
        for i in range(num_attributes):
            self.visited_attribute_counts[i] = 0

        self.visited_predicate_counts = {}
        for i in range(num_predicates):
            self.visited_predicate_counts[i] = 0

        self.visited_qtype_counts = {}
        for i in range(6):
            self.visited_qtype_counts[i] = 0

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Bilinear:
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            m.bias.data.fill_(0.)

    def assign_network_plasticity_all_params(self, require_grad=False):
        self.set_network_plasticity(self.prediction_network, require_grad=require_grad)
        self.set_network_plasticity(self.bbox_encoder, require_grad=require_grad)
        self.box_null.requires_grad = require_grad
        self.pred_null.requires_grad = require_grad
        self.attr_null.requires_grad = require_grad
        self.pred_encoder.weight.requires_grad = require_grad
        self.attr_encoder.weight.requires_grad = require_grad
        if not self.tied_target_weights:
            self.set_network_plasticity(self.bbox_encoder_target, require_grad=require_grad)
            self.pred_encoder_target.weight.requires_grad = require_grad
            self.attr_encoder_target.weight.requires_grad = require_grad

    def set_network_plasticity(self, net, require_grad=False):
        for module in net.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(require_grad)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(require_grad)

            if isinstance(module, nn.BatchNorm1d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(require_grad)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(require_grad)
                module.track_running_stats = require_grad
                module.eval()

    def make_encoding_network(self, shape, name="encoder_", final_activation='sigmoid'):
        encoder = nn.Sequential()
        encoders_ref = []  # in case we want to reference internal layers
        for i in range(len(shape) - 1):
            encoder_ref = name + str(i)
            if i == (len(shape) - 2):
                # final layer, so need to change activation function
                if final_activation == 'sigmoid':
                    encoder.add_module(encoder_ref,
                                       nn.Sequential(nn.Linear(shape[i], shape[i + 1]), self.norm(shape[i + 1]),
                                                     nn.Sigmoid()))
                elif final_activation == 'mish':
                    encoder.add_module(encoder_ref,
                                       nn.Sequential(nn.Linear(shape[i], shape[i + 1]), self.norm(shape[i + 1]),
                                                     Mish()))
                else:
                    raise NotImplementedError
            else:
                encoder.add_module(encoder_ref,
                                   nn.Sequential(nn.Linear(shape[i], shape[i + 1]), self.norm(shape[i + 1]),
                                                 Mish(), nn.Dropout(p=self.dropout_prob)))
                encoders_ref.append(encoder_ref)

                # initialize weights
                for e in encoder:
                    e.apply(self.init_weights)
        return encoder, encoders_ref

    def compute_bbox_embeddings(self, bbox_batch, target=False):
        if self.tied_target_weights or not target:
            bbox_feats = self.bbox_encoder(bbox_batch)
        else:
            bbox_feats = self.bbox_encoder_target(bbox_batch)
        return bbox_feats

    def compute_predicate_embeddings(self, pred_batch, target=False):
        if self.tied_target_weights or not target:
            pred_feats = self.pred_encoder(pred_batch)
        else:
            pred_feats = self.pred_encoder_target(pred_batch)
        return pred_feats

    def compute_attribute_embeddings(self, attr_batch, target=False):
        if self.tied_target_weights or not target:
            attr_feats = self.attr_encoder(attr_batch)
        else:
            attr_feats = self.attr_encoder_target(attr_batch)
        return attr_feats

    def get_target_features(self, bbox_batch, pred_batch, attr_batch, question_labels, device):
        # bbox_batch: N x box_feat_len
        # pred_batch: M x num_total_pred
        # attr_batch: P x num_total_attr
        # question_labels: num_questions x 2 (dim0: q_type, dim1: ix in batch array)

        # get embedded features
        bbox_feats = self.compute_bbox_embeddings(bbox_batch, target=True)
        pred_feats = self.compute_predicate_embeddings(pred_batch, target=True)
        attr_feats = self.compute_attribute_embeddings(attr_batch, target=True)

        master_features = torch.empty((len(question_labels), self.embedding_dimension)).to(device)
        for q_type in QType:
            q_ix = np.where(question_labels[:, 0] == q_type)[0]
            if len(q_ix) == 0:
                # no questions of this type, continue
                continue
            curr_questions = question_labels[q_ix]
            if q_type == QType.spos or q_type == QType.spas or q_type == QType.spoo:
                feats = bbox_feats[curr_questions[:, 1]]
            elif q_type == QType.spop or q_type == QType.spap:
                feats = pred_feats[curr_questions[:, 1]]
            elif q_type == QType.spaa:
                feats = attr_feats[curr_questions[:, 1]]
            else:
                raise NotImplementedError
            master_features[q_ix] = feats

        # return truth features for all questions in the batch
        if self.normalize_embeddings:
            master_features = F.normalize(master_features, p=2, dim=1)
        return master_features

    def forward(self, bbox_batch, pred_batch, attr_batch, question_labels, device):
        # bbox_batch: N x box_feat_len
        # pred_batch: M x num_total_pred
        # attr_batch: P x num_total_attr
        # question_labels: num_questions x 3 (dim0: q_type, dim1: ix in batch array, dim2: ix in batch array)

        # get embedded features
        bbox_feats = self.compute_bbox_embeddings(bbox_batch)
        pred_feats = self.compute_predicate_embeddings(pred_batch)
        attr_feats = self.compute_attribute_embeddings(attr_batch)

        # for each question type, get the appropriate features and put them into pre-initialized feature array
        master_features = torch.empty((len(question_labels), 3 * self.embedding_dimension)).to(device)
        for q_type in QType:
            q_ix = np.where(question_labels[:, 0] == q_type)[0]
            if len(q_ix) == 0:
                # no questions of this type, continue
                continue
            curr_questions = question_labels[q_ix]
            if q_type == QType.spos:
                feats1 = self.box_null.to(device).repeat(len(curr_questions), 1)
                feats2 = pred_feats[curr_questions[:, 1]]
                feats3 = bbox_feats[curr_questions[:, 2]]
            elif q_type == QType.spas:
                feats1 = self.box_null.to(device).repeat(len(curr_questions), 1)
                feats2 = pred_feats[curr_questions[:, 1]]
                feats3 = attr_feats[curr_questions[:, 2]]
            elif q_type == QType.spop:
                feats1 = bbox_feats[curr_questions[:, 1]]
                feats2 = self.pred_null.to(device).repeat(len(curr_questions), 1)
                feats3 = bbox_feats[curr_questions[:, 2]]
            elif q_type == QType.spap:
                feats1 = bbox_feats[curr_questions[:, 1]]
                feats2 = self.pred_null.to(device).repeat(len(curr_questions), 1)
                feats3 = attr_feats[curr_questions[:, 2]]
            elif q_type == QType.spoo:
                feats1 = bbox_feats[curr_questions[:, 1]]
                feats2 = pred_feats[curr_questions[:, 2]]
                feats3 = self.box_null.to(device).repeat(len(curr_questions), 1)
            elif q_type == QType.spaa:
                feats1 = bbox_feats[curr_questions[:, 1]]
                feats2 = pred_feats[curr_questions[:, 2]]
                feats3 = self.attr_null.to(device).repeat(len(curr_questions), 1)
            else:
                raise NotImplementedError

            concat_feats = torch.cat([feats1, feats2, feats3], dim=1)
            master_features[q_ix] = concat_feats

        # return network predictions for all questions in the batch
        feats = self.prediction_network(master_features)
        if self.normalize_embeddings:
            feats = F.normalize(feats, p=2, dim=1)
        return feats
