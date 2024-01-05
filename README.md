# Torch Implementation of FGVC Datasets and Baselines

This repository provides an implementation of the most renowned FGVC (Fine-Grained Visual Categorization) datasets, complete with a training script using ResNet50. Some datasets, originally unavailable or with altered structures, have been reorganized and are now accessible via Google Drive for automated downloading.

The purposes of this project are
- providing unified interfaces to FGVC datasets
- easy access to dataset using auto downloading before training
- expendable to new datasets

- The scripts have been tested on CUDA 11.8 with torch 2.0.

Datasets listed:

- [FGVC Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Caltech UCSD Birds (CUB-200-2011)](http://www.vision.caltech.edu/visipedia/CUB-200.html)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [North American Birds (NABirds)](https://dl.allaboutbirds.org/nabirds)
- [iNaturalist 2017(Inat2017)](https://github.com/visipedia/inat_comp/tree/master/2017)
- [TinyImageNet](https://www.kaggle.com/c/tiny-imagenet)

## Prerequisites
This project is compatible with both Linux and Windows operating systems.

#### Linux & Windows
```shell
# Clone the repository:
git clone https://github.com/qiyuliaogh/FGVC_baselines_pytorch.git
cd FGVC_baselines_pytorch

# Install dependencies: 
pip install torch numpy time tqdm albumentations torchvision json pandas cv2 random scipy matplotlib
```

Then modify the including in training script, for example, if you want to use Stanford dogs datasset, use:

```shell
from datasets.Dogs import Dogs as FGVC_Dataset
```
Uncomment this line and comment out the rest.

## Running the Training Script
#### Linux & Windows
```shell
python model_trainer.py --device cuda:0
```