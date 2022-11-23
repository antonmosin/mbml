"""Utility functions"""

import os
import random
import numpy as np
import torch


def sigmoid_range(x, low: float, high: float):
    """Sigmoid function with range `(low, high)`
    Source: https://github.com/fastai/fastai/blob/422fd1588e84704a15be02d99e8d64aab236a25f/fastai/layers.py#L99
    """
    return torch.sigmoid(x) * (high - low) + low

def worker_init_fn(worker_id):
    #https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

#g = torch.Generator()
#g.manual_seed(0)

def seed_torch(seed=0):
    """
    Ensure reproduceable results
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
