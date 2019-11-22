import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch import Tensor

data_path='../dataset/train/'

def load_npy(data):
    data=np.load(data_path+data)
    return data

class Style_Dataset(Dataset):
    def __init__(self, audio_data, label_data, transform=None):
        self.data=load_npy(audio_data)
        self.label=load_npy(label_data)
        self.avgv=load_npy('avg.npy')
        self.stdv=load_npy('std.npy')
        #self.transform = transform

    def __getitem__(self,index):
        audio=self.data[index]
        ## normaliztion
        audio = np.transpose(np.log(1+10000*audio)) ####?????????
        audio=(audio-self.avgv)*self.stdv
        audio_tensor=Tensor(audio).view(1,audio.shape[0],audio.shape[1]).float()
        
        classifi=int(self.label[index])

        return [audio_tensor,classifi]

    def __len__(self):
        return len(self.label)

data=Style_Dataset('data.npy','label.npy').__getitem__(200)
#print(Style_Dataset('data.npy','label.npy').__len__())
#print(data[0].size()[2:])
#print(data[0])