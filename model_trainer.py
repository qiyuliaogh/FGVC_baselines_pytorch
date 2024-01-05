import torch
import numpy as np
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torch.cuda.amp import autocast as autocast

from datasets.CUB_2011 import Cub2011 as FGVC_Dataset
# from datasets.Aircraft import Aircraft as FGVC_Dataset
from datasets.Dogs import Dogs as FGVC_Dataset
# from datasets.TinyImageNet import TinyImageNet as FGVC_Dataset
# from datasets.INat2017 import INat2017 as FGVC_Dataset
from datasets.Cars import Cars as FGVC_Dataset                        #needs manual download
from datasets.NABirds import NABirds as FGVC_Dataset                  #needs manual download

class NetworkManager(object):
    def __init__(self, options):
        self.options = options
        self.device = options['device']

        print('Starting to prepare network and data...')

        train_transform_list = [
            A.RandomResizedCrop(self.options['crop_size'], self.options['crop_size'], scale=(0.5,1)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()]

        test_transforms_list = [
            A.SmallestMaxSize(self.options['reshape_size']),
            A.CenterCrop(self.options['crop_size'], self.options['crop_size']),
            A.Normalize(),
            ToTensorV2()]

        train_dataset = FGVC_Dataset(root='./download', train=True, transform=A.Compose(train_transform_list, p=1.))
        test_dataset = FGVC_Dataset(root='./download', train=False, transform=A.Compose(test_transforms_list, p=1.))
        self.num_classes = train_dataset.num_class()

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.options['batch_size'], shuffle=False, num_workers=4, pin_memory=True
        )

        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.net = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay'])
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.solver, T_0=1,T_mult=2, eta_min=1e-5)

    def train(self):
        test_acc = list()
        train_acc = list()
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc\tLearningRate')
        print('-'*50)
        best_acc = 0.0
        for epoch in range(self.options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            self.net.train(True)
            for imgs, labels in tqdm(self.train_loader, desc="training", leave=False):
                self.solver.zero_grad()
                with autocast():
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    output = self.net(imgs)
                    loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()

            test_acc_epoch = self._accuracy()
            train_acc_epoch = num_correct.detach().cpu().numpy()*100. / num_total
            avg_train_loss_epoch  = sum(train_loss_epoch)/len(train_loss_epoch)
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            save_flg = ""
            if test_acc_epoch>=best_acc:
                best_acc = test_acc_epoch
                saved_parms = self.net.state_dict()
                torch.save(saved_parms, './models/saved_model.pkl')
                save_flg = "model saved!"
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%\t{:f}\t'.format(epoch, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch, self.solver.param_groups[0]['lr'])+time.strftime('%H:%M:%S\t',time.localtime(time.time()))+save_flg)

            self.schedule.step()
        return best_acc

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels in tqdm(self.test_loader, desc="testing", leave=False):
                with autocast():
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    output = self.net(imgs)
                _, pred = torch.max(output, 1)

                num_acc += torch.sum(pred==labels.detach_())
                num_total += labels.size(0)
        return num_acc.detach().cpu().numpy()*100./num_total

def main(base_lr=1e-3, weight_decay=1e-4, steps=5, batch_size=16, crop_size=256, reshape_size=224):
    parser = argparse.ArgumentParser(
        description='Options for base model finetuning on CUB_200_2011 datasets')
    parser.add_argument('--base_lr', type=float, default=base_lr)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=steps)
    parser.add_argument('--reshape_size', type=int, default=reshape_size)
    parser.add_argument('--crop_size', type=int, default=crop_size)
    args = parser.parse_args()
    assert args.gpu_id.__class__ == int


    options = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'reshape_size': args.reshape_size,
        'crop_size': args.crop_size,
        'device': torch.device('cuda:'+str(args.gpu_id))
    }

    manager = NetworkManager(options)
    return manager.train()

if __name__ == '__main__':
    print(main(base_lr=1e-2,weight_decay=1e-5,steps=7, batch_size=8, reshape_size=256, crop_size=224))