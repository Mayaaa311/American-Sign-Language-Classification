import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
from math import sqrt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, io, models, ops, transforms, utils
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import pandas as pd
from PIL import Image
# from torchvision import datasets, io, models, ops, transforms, utils
import os
# import data as dataset

def file_exists(filename):
    folders = ['avg_dev', 'avg_test', 'avg_train']
    filename = filename.replace('/','_')+".png"
    # print(filename)
    for folder in folders:
        if os.path.exists(os.path.join(folder, filename)):
            return True
    return False

class HandSignDataset(Dataset):
    def __init__(self, csv_file, root_dir, partition, transform=None):
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.df = self.df[self.df['partition'] == partition]
        self.df = self.df[self.df['filename'].apply(file_exists)]

    # define a function to check if a file exists in any of the folders

        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx, self.df.columns.get_loc('filename')]
        filename_img = self.df.iloc[idx, self.df.columns.get_loc('filename')].replace('/','_')
        label = self.df.iloc[idx, self.df.columns.get_loc('Label')]
        label = ord(label) - 97
        label = torch.tensor(label).long()
        # if label != 'a' and label != 'b' and label != 'c' and label != 'd':
        #     print(label)
        image_path = os.path.join(self.root_dir, filename_img+".png")
        bbox_path = os.path.join("BBox", filename, "0000.txt")
        
        try:
            with open(bbox_path) as f:
                bbox_info = f.readline().split(',')
                print("bbox_info",bbox_info)
            x0, y0, x1, y1, _ = bbox_info
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            image = Image.open(image_path).convert('RGB')
            image = image.crop((x0, y0, x1, y1))
        except FileNotFoundError:
            image = Image.open(image_path).convert('RGB')
        if(np.sum(image)==0):
            print('ALL ZERO')

        if self.transform:
            image = self.transform(image)
        # print('image name: ',filename_img," | shape:",image.shape," | label: ",label)
#         utils.save_image(img, f"/ImageOutput/{filename_img}_T.png")
        # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
#         save_image(img, filename +'_T.png')
        return image, label