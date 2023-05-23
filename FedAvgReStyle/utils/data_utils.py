import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform



import torch
from torchvision import transforms,datasets

import torch.utils.data as data

import numpy as np


def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  max_iter=1000):
    """
        Obtain sample index list for each client from the Dirichlet distribution.
        This LDA method is first proposed by :
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
        This can generate nonIIDness with unbalance sample number in each label.
        The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
        Dirichlet can support the probabilities of a K-way categorical event.
        In FL, we can view K clients' sample number obeys the Dirichlet distribution.
        For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
        Parameters
        ----------
            label_list : the label list from classification/segmentation dataset
            client_num : number of clients
            classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
            alpha: a concentration parameter controlling the identicalness among clients.
            task: CV specific task eg. classification, segmentation
        Returns
        -------
            samples : ndarray,
                The drawn samples, of shape ``(size, k)``.
    """

    print("Dataset partitions")

    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list)

    # guarantee the minimum number of sample in each client
    iter_counter = 0

    best_std = np.inf
    best_idx_batch = [[] for _ in range(client_num)]

    while iter_counter < max_iter:
        iter_counter += 1
        idx_batch = [[] for _ in range(client_num)]

        # for each classification in the dataset
        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(label_list == k)[0]
            idx_batch, std_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch,
                                                                                      idx_k)

        if std_size < best_std:
            best_std = std_size
            best_idx_batch = idx_batch.copy()
            print(f'Best std: {std_size}, iteration number: {iter_counter}')

    for i in range(client_num):
        np.random.shuffle(best_idx_batch[i])
        net_dataidx_map[i] = best_idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    std_size = np.std([len(idx_j) for idx_j in idx_batch])

    return idx_batch, std_size




def create_dataset_and_evalmetrix(args):
    if args.split_type == 'central':
        args.dis_csv_files = ['central']
        
    if args.dataset == 'cifar10':
        data_dir = '../data7cifar/'
        apply_transform = transforms.Compose(
            [transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])
        
        data_all = datasets.CIFAR10(data_dir,download=True, 
                                   transform=apply_transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
        
        """
                        TODO: Look the other code and ask 
        """
        
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []
        args.best_eval_loss[single_client] = 9999
        
        
        pass
    
    


class DatasetFLViT(data.Dataset):
    
    def __init__(self,args,phase):
        super(DatasetFLViT,self).__init__()
        self.phase = phase
        
        self.transfroms = transforms.Compose(
            [transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])
        
        #if args.phase = ""
        
    