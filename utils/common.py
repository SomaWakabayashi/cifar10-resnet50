import os
import random
import numpy as np
import torch

def fix_seed(seed: int = 42) -> None:
    """
    再現性のためにシードを固定する
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Seed fixed to {seed}")