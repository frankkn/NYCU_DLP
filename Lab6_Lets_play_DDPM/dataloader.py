import pandas as pd
from PIL import Image
from torch.utils import data
import torch
import json
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
from torchvision import transforms
from torchvision.utils import save_image


def getData(root, mode):
    obj_path = root + 'objects.json'
    with open(obj_path) as file:
        obj_dict = json.load(file)
    
    data_path = root + mode + '.json'
    with open(data_path) as file:
        data = json.load(file)

    lb = LabelBinarizer()
    lb.fit([i for i in range(24)])
    
    if mode == 'train':
        img_name = []
        labels = []
        for key, value in data.items():
            img_name.append(key) # "CLEVR_train_002066_2.png"
            tmp = []
            for i in range(len(value)): # ["cyan cube", "cyan sphere", "brown cylinder"]
                tmp.append(np.array(lb.transform([obj_dict[value[i]]]))) # "brown cylinder" -> 第2類 -> [0,0,...,0,,1,0,0]  
            labels.append((np.sum(tmp, axis=0))) # axis = 0, 對每個tmp的row vector進行element-wise的相加
        # print("train_img_name:", len(img_name)) # 18009
        # print("train_label:", len(label)) # 18009
        labels = torch.tensor(np.array(labels))
        # print("label[0]:", label[0])
        # label[0]: tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        return img_name, labels
    
    elif mode == 'test' or mode == 'new_test':
        labels = []
        for value in data:
            tmp = []
            for i in range(len(value)):
                tmp.append(np.array(lb.transform([obj_dict[value[i]]])))
            labels.append(np.sum(tmp, axis=0))
        # print("test_label:", len(label))
        labels = torch.tensor(np.array(labels))
        return labels

    

class iclevrLoader(data.Dataset):
    def __init__(self, root, mode, partial =1.0):
        self.root = root # ./dataset
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
            img_path = os.path.join(self.root, "iclevr", self.img_name[index])
            img = Image.open(img_path).convert('RGB')

            transform=transforms.Compose([
                transforms.Resize([64, 64]), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            img = transform(img)
        else:
            img = torch.ones(1)
        
        label = self.label[index]
        return img, label
    
def save_images(images, name):
    save_image(images, fp = "./"+name+".png")
    
# tester = iclevrLoader("./dataset/", "train")