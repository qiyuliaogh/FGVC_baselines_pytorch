## Torch implementation of FGVC datasets and their baselines

This project implements most famous FGVC datasets with training script using ResNet50.

As some of the datasets are not available online or structure has been changed, I reoganized the structure and upload them to google drive for auto downloading.

I've tested the loading and training scripts on CUDA11.8+torch2.0

The following libs are required:

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
and uncomment the rest, then,

#### Linux & Windows
```shell
# Run the training script.
python model_trainer.py --device cuda:0
```