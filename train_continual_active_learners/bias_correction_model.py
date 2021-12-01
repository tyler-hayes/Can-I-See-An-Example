import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from utils import CMA


class BiasCorrectionModel(nn.Module):
    """
    Parameters to correct bias on attribute and predicate classes due to train/test distribution mismatch
    """

    def __init__(self, num_predicates, num_attributes, lr=0.01, num_iters=500, use_alpha_predicate=False,
                 use_alpha_attribute=True, optimizer='lbfgs'):
        super(BiasCorrectionModel, self).__init__()

        # bias correction parameters
        self.predicate_biases = nn.Parameter(torch.ones(num_predicates))
        self.attribute_biases = nn.Parameter(torch.ones(num_attributes))
        if use_alpha_predicate:
            self.predicate_alpha = nn.Parameter(torch.ones(num_predicates))
        if use_alpha_attribute:
            self.attribute_alpha = nn.Parameter(torch.ones(num_attributes))

        # training settings
        self.lr = lr
        self.num_iters = num_iters
        self.use_alpha_predicate = use_alpha_predicate
        self.use_alpha_attribute = use_alpha_attribute
        self.optimizer = optimizer

    def forward(self, dists, type):
        return self.correct_biases(dists, type)

    def correct_biases(self, dists, type):
        """
        Perform bias correction
        """
        if type == 'predicates':
            if self.use_alpha_predicate:
                dists = self.predicate_alpha * dists + self.predicate_biases
            else:
                dists += self.predicate_biases
        elif type == 'attributes':
            if self.use_alpha_attribute:
                dists = self.attribute_alpha * dists + self.attribute_biases
            else:
                dists += self.attribute_biases
        return dists

    def train_with_adam(self, dists, labels, criterion, type):
        print('\nTraining with Adam...')
        msg = '\rEpoch (%d/%d) -- train_loss=%1.4f'
        tensor_dset = TensorDataset(dists, labels)
        data_loader = DataLoader(tensor_dset, batch_size=256, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader))

        for e in range(self.num_iters):
            total_loss = CMA()
            for i, (dists_batch, truth_batch) in enumerate(data_loader):
                loss = criterion(self(dists_batch, type), truth_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())
            print(msg % (e + 1, self.num_iters, total_loss.avg), end="")
            scheduler.step()

    def train_with_lbfgs(self, dists, labels, criterion, type):
        print('\nTraining with LBFGS...')
        optimizer = optim.LBFGS(self.parameters(), lr=self.lr, max_iter=self.num_iters)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self(dists, type), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

    # This function probably should live outside of this class
    def train_bias_correction_parameters(self, dists, truths, type):
        """
        Train the bias correction parameters
        """
        print('\nTraining bias correction model...')
        self.train()
        nll_criterion = nn.CrossEntropyLoss()

        # collect all the dists and labels into dataloader
        dists = torch.from_numpy(np.concatenate(dists, axis=0))
        labels = torch.from_numpy(np.concatenate(truths, axis=0))
        labels = torch.argmax(labels, dim=1)

        # calculate NLL before bias correction
        before_bc_nll = nll_criterion(dists, labels).item()
        print('Before bias correction - NLL: %.3f' % before_bc_nll)

        # optimize the bias correction parameters w.r.t. NLL
        if self.optimizer == 'lbfgs':
            self.train_with_lbfgs(dists, labels, nll_criterion, type)
        else:
            self.train_with_adam(dists, labels, nll_criterion, type)

        # calculate NLL after bias correction
        after_bc_nll = nll_criterion(self(dists, type), labels).item()
        print('\nAfter bias correction - NLL: %.3f' % after_bc_nll)
