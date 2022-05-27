# utils functions
import numpy as np 
import random
import os
from time import time
import pickle
import pdb
import json
import torch
import logging

def setup_seed(seed):
    '''
    seed: the input random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_random_dir_name():
    '''
    generating the dir name using the datetime
    '''
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


def construct_log(args):
    '''
    define logger for recording all experimental results
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    os.makedirs(args.log_dir, exist_ok = True)
    handler = logging.FileHandler(os.path.join(args.log_dir, "log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    if not args.auto_deploy:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
    return logger