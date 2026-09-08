from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from PIL import Image

class CustomXRayDataset(Dataset):
    def __init__(self,img_dir='None',label_dir='None',transforms='None'):
        df = pd.read_csv(label_dir)
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small','chexpert')
        
        labels_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        self.paths = df['Path'].to_numpy(dtype=object)
        self.labels = df[labels_name].to_numpy(dtype='float32')
        
        self.img_dir = img_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        pth = os.path.join(self.img_dir,self.paths[idx])
        with Image.open(pth) as img:
            img = img.convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
        label = torch.from_numpy(self.labels[idx])
        return img,label
    