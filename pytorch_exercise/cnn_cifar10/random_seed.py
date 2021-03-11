import random
import torch
import numpy as np

def make_random(seed_numder) :
    seed_number = 42
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)