import os
import pandas as pd
from torchvision.datasets.utils import download_url
import torch
import numpy as np
import scipy.io as sio
from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg, list_dir
np.random.seed(0)
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import sys
import json
import warnings
from random import choice
import random
from scipy.io import loadmat
from google_drive_downloader import GoogleDriveDownloader as gdd


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}
def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names

def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents

class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root="./download/", train=True, transform=None, target_transform=None, download=True):
        super(NABirds, self).__init__(root=root, transform=transform, target_transform=target_transform)
        if download is True:
            self.download()

        print("Data: NABirds: "+"training" if train else "testing")
        dataset_path = os.path.join(root, 'nabirds')
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.loader = default_loader
        self.train = train

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def num_class(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, target

    def download(self):
        if os.path.exists(join(self.root, 'nabirds')):
            print('Files already downloaded and verified')
            return

        url = 'https://drive.google.com/file/d/18ePqj0GXpiUG6bOqw_82oo4a63RZZVP9/view?usp=sharing'
        download_url(url, self.root, 'nabirds.tar.gz')
        print('Extracting downloaded file: ' + join(self.root, 'nabirds.tar.gz'))

        extract_archive(join(self.root, 'nabirds.tar.gz'))
