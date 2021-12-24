# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image
import torchvision
import torch
from sklearn.preprocessing import LabelEncoder
from .nabirds_dataloader import ImageList, make_dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
        return pil_loader(path)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
    

def get_data(root, n_labeled, num_classes, transform_train=None, transform_val=None, download=True):

    train_root = root + 'nabirds_train.txt'
    test_root = root + 'nabirds_test.txt'
    print('train and test roots', train_root, test_root)
    
    base_dataset_list = ImageList(open(train_root).readlines(), transform=transform_train)
    
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        base_dataset_list.targets, n_labeled, num_classes)
    
    train_labeled_dataset_list = NAB_labeled(train_labeled_idxs, 
                         open(train_root).readlines(), labeled=True,
                         transform=transform_train)
    
    train_unlabeled_dataset_list = NAB_labeled(train_unlabeled_idxs,
                         open(train_root).readlines(), labeled=False,
                         transform=TransformTwice(transform_train))
    
    val_dataset_list = NAB_labeled(val_idxs, open(train_root).readlines(),
                                   labeled=True, transform=transform_val)
    
    test_dataset_list = ImageList(open(test_root).readlines(), transform=transform_val)
    
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset_list, train_unlabeled_dataset_list, val_dataset_list, test_dataset_list


def train_val_split(labels, n_labeled_per_class, num_classes):
    """
    Creates training labeled, unlabeled, and validation splits with
    equal samples of each class.
    """
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    # Use 20% labaled samples per class. 

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        n_labeled_per_class = math.ceil(0.20*len(idxs))
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[:-n_labeled_per_class])
        val_idxs.extend(idxs[-n_labeled_per_class:])
    
    print(f'#Train Samples {len(train_labeled_idxs)} | #Unlabeled Samples {len(train_unlabeled_idxs)} | #Validation Samples {len(val_idxs)}')
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

nab_mean = (125.30513277, 129.66606421, 118.45121113)
nab_std = (57.0045467, 56.70059436, 68.44430446)

def normalise(x, mean=nab_mean, std=nab_std):
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


class NAB_labeled(object):

    def __init__(self, indexs, image_list, labeled=True, labels=None, transform=None,
                 target_transform=None, loader=default_loader):
        imgs, targets = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = np.array(imgs)[indexs]
        self.label_encoder = LabelEncoder().fit(targets)
        if labeled == True:
            self.targets = np.array(self.label_encoder.transform(targets))[indexs]
        elif labeled == False:
            self.targets = np.array([-1 for i in range(len(targets))])[indexs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index]
        path = '/home/ashimag/Datasets/nabirds_A100/nabirds/images/' + path
        img = self.loader(path)
        target = self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def get_label_encoder(self):
        return self.label_encoder
