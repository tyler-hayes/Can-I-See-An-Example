import errno
import os
import numpy as np
from enum import IntEnum
from torch.utils.tensorboard import SummaryWriter


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return list(rand_vals.keys())


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


def bool_flag(s):
    if s == '1' or s == 'True' or s == 'true':
        return True
    elif s == '0' or s == 'False' or s == 'false':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0/1 or True/False or true/false)'
    raise ValueError(msg % s)


class TFSubLogger:
    def __init__(self, parent_logger, prefix):
        self.parent_logger = parent_logger
        self.prefix = prefix

    def add_scalar(self, name, value, iteration):
        self.parent_logger.add_scalar(self.prefix + "/" + name, value, iteration)

    def message(self, message, name=""):
        self.parent_logger.message(message, self.prefix + "/" + name)


class Logger:
    def add_scalar(self, name, value, iteration):
        raise NotImplementedError

    def add_scalars(self, name, value, iteration):
        raise NotImplementedError

    def close(self):
        pass

    def get_logger(self, name):
        raise NotImplementedError

    def message(self, message, name=""):
        print("[" + name + "] " + message)


class TFLogger(SummaryWriter, Logger):
    def __init__(self, log_dir=None, verbose=False, **args):
        SummaryWriter.__init__(self, log_dir=log_dir)
        self.verbose = verbose

    def add_scalar(self, name, value, iteration):
        if self.verbose:
            print("[LOG]: At " + str(iteration) + ": " + name + " = " + str(value))
        SummaryWriter.add_scalar(self, name, value, iteration)

    def add_scalars(self, name, value, iteration):
        if self.verbose:
            print("[LOG]: At " + str(iteration) + ": " + name + " = " + str(value))
        SummaryWriter.add_scalars(self, name, value, iteration)

    def get_logger(self, name):
        return TFSubLogger(self, name)
