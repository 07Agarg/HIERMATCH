# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:42:55 2021

@author: Ashima
"""

import numpy as np
import torch
import torch.nn as nn

trees = [
	[0, 4, 5],
    [1, 1, 6],
    [2, 14, 3],
    [3, 8, 3],
    [4, 0, 6],
    [5, 6, 0],
    [6, 7, 4],
    [7, 7, 4],
    [8, 18, 7],
    [9, 3, 0],
    [10, 3, 0],
    [11, 14, 3],
    [12, 9, 1],
    [13, 18, 7],
    [14, 7, 4],
    [15, 11, 3],
    [16, 3, 0],
    [17, 9, 1],
    [18, 7, 4],
    [19, 11, 3],
    [20, 6, 0],
    [21, 11, 3],
    [22, 5, 0],
    [23, 10, 2],
    [24, 7, 4],
    [25, 6, 0],
    [26, 13, 4],
    [27, 15, 4],
    [28, 3, 0],
    [29, 15, 4],
    [30, 0, 6],
    [31, 11, 3],
    [32, 1, 6],
    [33, 10, 2],
    [34, 12, 3],
    [35, 14, 3],
    [36, 16, 3],
    [37, 9, 1],
    [38, 11, 3],
    [39, 5, 0],
    [40, 5, 0],
    [41, 19, 7],
    [42, 8, 3],
    [43, 8, 3],
    [44, 15, 4],
    [45, 13, 4],
    [46, 14, 3],
    [47, 17, 5],
    [48, 18, 7],
    [49, 10, 2],
    [50, 16, 3],
    [51, 4, 5],
    [52, 17, 5],
    [53, 4, 5],
    [54, 2, 5],
    [55, 0, 6],
    [56, 17, 5],
    [57, 4, 5],
    [58, 18, 7],
    [59, 17, 5],
    [60, 10, 2],
    [61, 3, 0],
    [62, 2, 5],
    [63, 12, 3],
    [64, 12, 3],
    [65, 16, 3],
    [66, 12, 3],
    [67, 1, 6],
    [68, 9, 1],
    [69, 19, 7],
    [70, 2, 5],
    [71, 10, 2],
    [72, 0, 6],
    [73, 1, 6],
    [74, 16, 3],
    [75, 12, 3],
    [76, 9, 1],
    [77, 13, 4],
    [78, 15, 4],
    [79, 13, 4],
    [80, 16, 3],
    [81, 19, 7],
    [82, 2, 5],
    [83, 4, 5],
    [84, 6, 0],
    [85, 19, 7],
    [86, 5, 0],
    [87, 5, 0],
    [88, 8, 3],
    [89, 19, 7],
    [90, 18, 7],
    [91, 1, 6],
    [92, 2, 5],
    [93, 15, 4],
    [94, 6, 0],
    [95, 0, 6],
    [96, 17, 5],
    [97, 8, 3],
    [98, 14, 3],
    [99, 13, 4]]

def get_order_family_target_(target, i):
    if target == -1:
        return target
    return trees[target][i]

def get_order_target(targets, level):
    order_target_list = []
    for i in range(targets.size(0)):
        target = torch.argmax(targets[i])
        order_target_list.append(trees[target][level+1])
    
    return np.array(order_target_list)

def get_order_family_target(targets):
    order_target_list = []
    family_target_list = []
    for i in range(targets.size(0)):
        order_target_list.append(trees[targets[i]][1])
        family_target_list.append(trees[targets[i]][2])

    order_target_list = torch.from_numpy(np.array(order_target_list))
    family_target_list = torch.from_numpy(np.array(family_target_list))

    return order_target_list, family_target_list