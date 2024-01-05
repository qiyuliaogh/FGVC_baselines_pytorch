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

class Cars(Dataset):
    def __init__(self, root="./download/", train=True, transform=None, download=True):
        print("Data: Cars: "+"training" if train else "testing")
        self.num_classes = 196
        self.root=root
        self.datapath = os.path.join(root, 'Cars/')

        if download is True:
            self.download()

        if train:
            list_path = os.path.join(self.datapath, 'cars_train_annos.mat')
            self.image_path = os.path.join(self.datapath, 'cars_train')
        else:
            list_path = os.path.join(self.datapath, 'cars_test_annos_withlabels.mat')
            self.image_path = os.path.join(self.datapath, 'cars_test')

        list_mat = loadmat(list_path)
        self.images = [f.item() for f in list_mat['annotations']['fname'][0]]
        self.labels = [f.item() for f in list_mat['annotations']['class'][0]]

        # transform
        self.transform = transform

    def __getitem__(self, item):
        path = os.path.join(self.image_path, self.images[item])
        target = self.labels[item] - 1
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, target

    def __len__(self):
        return len(self.images)

    def num_class(self):
        return self.num_classes

    def download(self):

        if os.path.exists(self.datapath):
            print('Files already downloaded and verified')
            return

        url = 'https://drive.google.com/file/d/1WjVF3bQUAic3Jgca_N3q2o_p4FCW7xe2/view?usp=drive_link'
        download_url(url, self.root, 'Cars.zip')
        print('Extracting downloaded file: ' + join(self.root, 'Cars.zip'))

        extract_archive(join(self.root, 'Cars.zip'))
