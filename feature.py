from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import transforms
import torch
import os
from dataset import CustomXRayDataset

#Train on kaggle
BASE_DIR = '/kaggle/input/datasets/ashery'

def get_dataloader():
    data_transform = {'train' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5030,0.5030,0.5030],[0.2893,0.2893,0.2893])
    ]),
                    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5030,0.5030,0.5030],[0.2893,0.2893,0.2893])
    ])}
    
    dataset = {x : CustomXRayDataset(BASE_DIR,os.path.join(BASE_DIR,'chexpert',f"{x}.csv"),data_transform[x])
          for x in ['train','valid']}
    
    dataloader = {
        'train' : DataLoader(dataset['train'],batch_size=64,shuffle=False,sampler=DistributedSampler(dataset['train']),num_workers=1,pin_memory=torch.accelerator.is_available(),persistent_workers=True,prefetch_factor=2),
        'valid' : DataLoader((dataset['valid']),batch_size=32,shuffle=False,sampler=DistributedSampler(dataset['valid'],shuffle=False),num_workers=1,pin_memory=torch.accelerator.is_available(),persistent_workers=True,prefetch_factor=2)
    }
    
    return dataloader
