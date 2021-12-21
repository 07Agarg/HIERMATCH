import numpy as np
from PIL import Image

import torchvision
import torch
# from cifar100_get_tree_target_2 import *
from cifar100_get_tree_target_3 import *

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar100(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):

    base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, n_labeled, 100)

    train_labeled_dataset = [CIFAR100_labeled(root, idxs, train=True, transform=transform_train, hierarchy=i) for i, idxs in enumerate(train_labeled_idxs)]
    
    train_unlabeled_dataset = [CIFAR100_unlabeled(root, idxs, train=True, transform=TransformTwice(transform_train)) for idxs in train_unlabeled_idxs]
    
    val_dataset = CIFAR100_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True, hierarchy=0)

#     print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(labels, n_labeled, n_classes):
    labels = np.array(labels)
    
    train_labeled_idxs = [[] for _ in range(len(n_labeled))]
    train_unlabeled_idxs = [[] for _ in range(len(n_labeled))]

    val_idxs = []

    for i in range(n_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        
        curr = 0
        for j in range(len(n_labeled)): # iterate over hierarchy, j = 0 is finest
            n_labeled_per_class = int(n_labeled[j]/n_classes)
            
            for k in range(j, len(n_labeled)): # for hierarchy above and including this, add idxs to labeled set
                train_labeled_idxs[k].extend(idxs[curr : curr + n_labeled_per_class])
            
            for k in range(j): # for hierarchy below this, add labeled samples as unlabeled samples
                train_unlabeled_idxs[k].extend(idxs[curr : curr + n_labeled_per_class])
            
            curr += n_labeled_per_class
        
        for j in range(len(n_labeled)):
            train_unlabeled_idxs[j].extend(idxs[curr : -50])
        
        val_idxs.extend(idxs[-50:])
    
    for i in range(len(n_labeled)):
        print(f'--------- Shape of training  set at hierarchy {i + 1} -> {len(train_labeled_idxs[i])} ---------')
        print(f'--------- Shape of unlabeled set at hierarchy {i + 1} -> {len(train_unlabeled_idxs[i])} ---------')
        
        np.random.shuffle(train_labeled_idxs[i])
        np.random.shuffle(train_unlabeled_idxs[i])
    
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar100_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar100_mean, std=cifar100_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, hierarchy=0):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))
        self.hierarchy = hierarchy
        
        print(f'Creating labeled dataset for hierarchy {self.hierarchy} | #Samples -> {len(self.data)}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = get_order_family_target_(target, self.hierarchy)
        return img, target
    

class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        print(f'Creating unlabeled data of size {len(self.targets)}')
        
