import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fixes random seeds across python, numpy, and torch (CPU + CUDA), and
    enables deterministic cuDNN kernels. This is what makes a reported
    metric reproducible run-to-run rather than a lucky roll.

    Note: `cudnn.deterministic = True` can slow training slightly on GPU --
    an intentional tradeoff for reproducibility on a portfolio project.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False