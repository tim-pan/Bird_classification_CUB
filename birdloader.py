import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
'''
in loader part
I try many different combinations of data argumentation
the following is an example

trans_tr is data argumentation for training data
trans_te is data argumentation for testing and validating data
'''
trans_tr = transforms.Compose([
    # transforms.RandomResizedCrop(256),
    transforms.Resize(300),
    transforms.RandomCrop([256, 256]),
    transforms.Resize([224, 224]),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.01, 0.01), ratio=(1, 1))
])
trans_te = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop([224, 224]),
    # transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#we use a dictionary to combine the two trans
trans = {
    'train': trans_tr,
    'test':trans_te
}

def getData(mode):
    '''
    this function aims to list all the filenames in training, validating, and testing data
    *input:
    mode:'train' or 'dev' or 'test'
    *output:
    out number of data and list of data
    for example:
        mode == 'train': [['0001.jpg', '0012.jpg', '0323.jpg'], ['2. blue_bird', '2. blue_bird', '1. red_bird']], 2700
        mode == 'dev' : [['0001.jpg', '0012.jpg', '0323.jpg'], ['2. blue_bird', '2. blue_bird', '1. red_bird']], 300
        mode == 'test' : ['0123.jpg', '4563.jpg', '1203.jpg'], 3300

    '''
    if mode == 'train':
        with open('./dataset/train_new.txt') as fp:
            final = []
            str_li = fp.read().split()
            le = len(str_li)
            final.append(str_li[0: le: 2])
            final.append(str_li[1: le: 2])
            return final, int(le/2)
    elif mode == 'dev':
        with open('./dataset/dev_new.txt') as fp:
            final = []
            str_li = fp.read().split()
            le = len(str_li)
            final.append(str_li[0: le: 2])
            final.append(str_li[1: le: 2])
            return final, int(le/2)
    elif mode == 'test':
        with open('./dataset/testing_img_order.txt') as fp:
            str_li = fp.read().split()
            return str_li, len(str_li)
    else:
        raise ValueError('your mode is illegal')
        
class BirdImage(Dataset):
    '''
    *input:
    mode: your phase, 'train', 'dev', or 'test'
    '''
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.datainfo, self.le = getData(self.mode)
        with open('./dataset/classes.txt') as fp:
            li = fp.read().split()
            self.classes=dict(zip(li, range(0, 200)))

    def __getitem__(self, index):
        '''
        if mode == 'train': generator will generate image tensor(after data argumentation), and its label(number, instead of, for eg: '1. red_bird' )
        mode == 'dev': the same as above
        mode == 'test': generator will generate image tensor(after data argumentation), NO label
        '''
        if self.mode == 'train':
            img = Image.open(f'./dataset/{self.mode}/{self.datainfo[0][index]}')
            img = trans['train'](img)
            label = self.classes[self.datainfo[1][index]]
            return img, label

        elif self.mode == 'dev':
            img = Image.open(f'./dataset/train/{self.datainfo[0][index]}')
            img = trans['test'](img)
            label = self.classes[self.datainfo[1][index]]
            return img, label

        else:
            img = Image.open(f'./dataset/{self.mode}/{self.datainfo[index]}')
            img = trans['test'](img)
            return img

    def __len__(self):
        return self.le