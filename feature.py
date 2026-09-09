from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import transforms
import torch
import os
from dataset import CustomXRayDataset

#Train on kaggle
BASE_DIR = '/kaggle/input/datasets/ashery'

def get_dataloader(u_policy):
    data_transform = {'train' : transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation((-10,10)),
        transforms.RandomCrop(224),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1,1.0))],p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5030,0.5030,0.5030],[0.2893,0.2893,0.2893])
    ]),
                    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5030,0.5030,0.5030],[0.2893,0.2893,0.2893])
    ])}
    
    dataset = {x : CustomXRayDataset(BASE_DIR,os.path.join(BASE_DIR,'chexpert',f"{x}.csv"),data_transform[x],u_policy=u_policy)
          for x in ['train','valid']}
    
    dataloader = {
        'train' : DataLoader(dataset['train'],batch_size=64,shuffle=False,sampler=DistributedSampler(dataset['train']),num_workers=1,pin_memory=False),
        'valid' : DataLoader((dataset['valid']),batch_size=32,shuffle=False,sampler=DistributedSampler(dataset['valid'],shuffle=False),num_workers=1,pin_memory=False)
    }
    
    return dataloader
