import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    # Set PYTHONHASHSEED first
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    