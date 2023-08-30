import pandas as pd
from PIL import Image
from torch.utils import data
import torch
import json
import numpy as np
import os
from torchvision import transforms
from utils import label_to_onehot


def getData(root, mode):
    label_path = root + 'objects.json'
    with open(label_path) as file:
        label_dict = json.load(file)
    
    data_path = root + mode + '.json'
    with open(data_path) as file:
        data_dict = json.load(file)
    
    if mode == 'train':
        img_name = []
        img_labels = []
        for name, labels in data_dict.items():
            img_name.append(name) # name = "CLEVR_train_002066_2.png"
            img_labels.append([label_dict[i] for i in labels]) # labels = ["cyan cube", "cyan sphere", "brown cylinder"]
        return img_name, img_labels
    
    elif mode == 'test' or mode == 'new_test':
        labels = torch.zeros(len(data_dict), len(label_dict)) # 第一維:樣本數量，第二維:類別數量
        for i, objs in enumerate(data_dict):
            for obj in objs:
                labels[i, int(label_dict[obj])] = 1.
        return labels

    

class iclevrLoader(data.Dataset):
    def __init__(self, root, mode, partial=1.0):
        self.root = root # ./dataset/
        self.mode = mode
        self.partial = partial
        
        if mode == 'train':
            self.img_name, self.label = getData(root, mode)
        elif mode == 'test' or mode == 'new_test':
            self.label = getData(root, mode)  

    def __len__(self):
        """'return the size of dataset"""
        return int(len(self.label) * self.partial)

    def __getitem__(self, index):        
        if self.mode == 'train':

            img_path = os.path.join(self.root, "iclevr/", self.img_name[index])
            img = Image.open(img_path).convert('RGB')

            transform=transforms.Compose([
                transforms.Resize([64, 64]), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            img = transform(img)
        else:
            img = torch.ones(1)
        
        label = self.int2one_hot(self.label[index])
        return img, label
    
    def int2one_hot(self, int_list):
        one_hot = torch.zeros(24) # num of classes
        for i in int_list:
            one_hot[i] = 1.
            
        return one_hot
    
# tester = iclevrLoader("./dataset/", "test")