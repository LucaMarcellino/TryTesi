import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as torch_models
from efficientnet_pytorch import EfficientNet
from models import build_model
from torch.nn import Linear

def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log

    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)

def initialization_config(args, vit = False):
    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")
    
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.dataset == 'cifar10':
        args.num_classes = 10
    else:
        args.num_classes = 2
        
    if "ViT" in args.FL_platform:
        if 'tiny' in args.net_name:
            print('We use ViT tiny')
            from timm.models.vision_transformer import vit_tiny_patch16_224
            model = vit_tiny_patch16_224(pretrained=args.Pretrained)
        elif 'small' in args.net_name:
            print('We use ViT small')
            from timm.models.vision_transformer import vit_small_patch16_224
            model = vit_small_patch16_224(pretrained=args.Pretrained)
        else:
            from timm.models.vision_transformer import vit_base_patch16_224
            print('We use default ViT settting base')
            model = vit_base_patch16_224(pretrained=args.Pretrained)

        model.head = Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    # set output parameters
    print(args.optimizer_type)
    args.name = args.net_name + '_' + args.split_type + '_lr_' + str(args.learning_rate) + '_Pretrained_' \
                + str(args.Pretrained) + "_optimizer_" + str(args.optimizer_type) +  '_WUP_'  + str(args.warmup_steps) \
                + '_Round_' + str(args.max_communication_rounds) + '_Eepochs_' + str(args.E_epoch) + '_Seed_' + str(args.seed)


    # args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name)
    args.output_dir = os.path.join(args.output_dir, args.FL_platform, args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model
    