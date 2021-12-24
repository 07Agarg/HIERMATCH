# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:13:13 2021

@author: Ashima
"""
"""To create tree list like Flamingo's code, use file semantic-embeddings/class-hierarchy.py"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle

def get_order_family_target_(target, i):
    if target == -1:
        return target
    
    save_path = './nabirds_tree_list.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)
    
    return trees[target][i]


def get_order_family_target(targets):

    save_path = './nabirds_tree_list.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    order_target_list = []
    family_target_list = []

    for i in range(targets.size(0)):

        order_target_list.append(trees[targets[i]][2])
        family_target_list.append(trees[targets[i]][1])

    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)))
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)))

    return targets, family_target_list, order_target_list #from fine to coarse