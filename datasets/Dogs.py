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


class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

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
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root="./download/", train=True, transform=None, target_transform=None, download=True):
        super(Dogs, self).__init__(root=join(root, 'Dogs/'), transform=transform, target_transform=target_transform)

        print("Data: Dogs: "+"training" if train else "testing")
        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def num_class(self):
        return 120

    def __getitem__(self, index):

        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, target

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            # os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = sio.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = sio.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts
