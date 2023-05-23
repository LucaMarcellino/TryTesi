# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.option import args_parser
from utils.start_config import initialization_config
from utils.data_utils import create_dataset_and_evalmetrix


def train(args,model):
    os.makedirs(args.output_dir, exist_ok=True)
    
    pass

def main():
    args = args_parser()
    model = initialization_config(args)
    #writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    
    train(args,model)
    
if __name__ == '__main__':
    main()