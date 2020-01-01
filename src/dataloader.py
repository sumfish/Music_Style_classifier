import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch import Tensor
from utils import *

npy_dir='../dataset/'
#npy_dir='../dataset/cluster_npy/'   #########draw cluster

class Style_Dataset(Dataset):
    def __init__(self, path, audio_data, label_data, transform=None):
        self.transform = transform
        self.data=load_npy(npy_dir+path+audio_data)
        self.label=load_npy(npy_dir+path+label_data)
        #self.avgv=load_npy(path+'avg.npy')
        #self.stdv=load_npy(path+'std.npy') 

    def __getitem__(self,index):
        audio=self.data[index]
        #print(audio.shape) #(128,Time) 
        
        if self.transform is not None:
            audio=self.transform(audio) #is already float data
        else:
            audio=Tensor(audio).view(1,audio.shape[0],audio.shape[1])
        #print(audio.shape) # torch.Size([1, 128, 130])
        classifi=int(self.label[index]) #'01'->1
        return [audio,classifi]

    def __len__(self):
        return len(self.data)

'''
for i in range(Style_Dataset('data_2s/', 'data.npy','label.npy').__len__()):
    Style_Dataset('data_1s/', 'data.npy','label.npy').__getitem__(i)
'''
#k=Style_Dataset('data_1s_4class/', 'data.npy','label.npy').__len__()
#print(Style_Dataset('data.npy','label.npy').__len__())
#print(data[0].size()[2:])
#print(data[0])
'''
train_mean, train_std=compute_mean_std()
img_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(train_mean,),std=(train_std,))
])
# train & validation dataset
print(Style_Dataset('train/data_1s/','data.npy','label.npy', transform = img_transform).__getitem__(600))
print(Style_Dataset('train/data_1s/','data.npy','label.npy').__getitem__(600))
'''