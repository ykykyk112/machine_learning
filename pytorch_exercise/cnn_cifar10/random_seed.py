import random
import torch
import numpy as np

def fix_randomness(seed_numder) :
    seed_number = 42
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_number)
    random.seed(seed_number)