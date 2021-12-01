import math
import os
import sys

import torch
from torch import nn, optim

from train_continual_active_learners.triple_completion_utils import get_scores_and_truth, compute_val_loss
from train_continual_active_learners.vg_qtype_dset_loader_utils import build_loaders
from utils import CMA


def get_pre_train_data_loaders(args, drop_last=False):
    # standard data loaders
    vocab, train_loader, unlabeled_train_loader, pre_train_labeled_loader, val_loader, test_loader, class_ix = build_loaders(
        args, drop_last=drop_last)

    return vocab, train_loader, val_loader, test_loader, class_ix


def pre_train(args, model, data_loader, test_loader, logger, device, verbose=True, inc=None, epochs=None,
              iteration_level=False):
    if epochs is None:
        epoch = args.epochs
    else:
        epoch = epochs

    print('\nStarting training...')

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd, momentum=0.9)

    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader))
    else:
        raise NotImplementedError

    # setup model
    model = model.to(device)
    model.train()

    msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- train_loss=%1.4f'
    msg_cluster = '\nEpoch (%d/%d) -- train_loss=%1.4f'

    for e in range(epoch):
        total_loss = CMA()

        for i, batch in enumerate(data_loader):

            scores, truth = get_scores_and_truth(args, model, batch, device)
            loss = criterion(scores, truth)

            if math.isnan(loss.item()):
                print('\nEncountered NaN loss and exiting.')
                sys.exit()

            # update network
            optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())

            if verbose:
                print(msg % (e + 1, epoch, i + 1, len(data_loader), total_loss.avg), end="")

            if iteration_level and (i % args.ckpt_epoch == 0):
                # compute val loss
                val_loss = compute_val_loss(args, model, test_loader, criterion, device)

                # save losses for all increments
                if inc is not None:
                    logger_name = 'loss_%d' % inc
                else:
                    logger_name = 'loss'

                logger.add_scalar(logger_name + '/train_iter', total_loss.avg, i)
                logger.add_scalar(logger_name + '/val_iter', val_loss, i)

                # save ckpt every args.ckpt_epoch epochs
                d = {
                    'iter': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.avg,
                }
                if inc is None:
                    name = 'triple_completion_model_iter_%d_ckpt.pth' % i
                else:
                    name = 'triple_completion_model_iter_%d_inc_%d_ckpt.pth' % (i, inc)

                torch.save(d, os.path.join(args.output_dir, name))

        if verbose:
            print(msg_cluster % (e + 1, epoch, total_loss.avg))

        scheduler.step()

        # compute val loss
        val_loss = compute_val_loss(args, model, test_loader, criterion, device)

        # save losses for all increments
        if inc is not None:
            logger_name = 'loss_%d' % inc
        else:
            logger_name = 'loss'

        logger.add_scalar(logger_name + '/train', total_loss.avg, e)
        logger.add_scalar(logger_name + '/val', val_loss, e)

        if (e + 1) % args.ckpt_epoch == 0:
            # save ckpt every args.ckpt_epoch epochs
            d = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.avg,
            }
            if inc is None:
                name = 'triple_completion_model_epoch_%d_ckpt.pth' % e
            else:
                name = 'triple_completion_model_epoch_%d_inc_%d_ckpt.pth' % (e, inc)

            torch.save(d, os.path.join(args.output_dir, name))

    # save final ckpt
    d = {
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.avg,
    }
    if inc is None:
        name = 'triple_completion_model_final_ckpt.pth'
    else:
        name = 'triple_completion_model_final_inc_%d_ckpt.pth' % inc
    torch.save(d, os.path.join(args.output_dir, name))
