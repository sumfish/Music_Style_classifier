import torch.nn as nn
import torch
import torch.nn.functional as F

## each cnn is zero-padded
class S_Model(torch.nn.Module):
    def __init__(self):
        super(S_Model,self).__init__()
        self.conv1= nn.Conv2d(1,32, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(32,32, kernel_size=3, padding=1)
        self.conv3= nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.conv4= nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.conv5= nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.conv6= nn.Conv2d(128,128, kernel_size=3, padding=1)
        #self.conv7= nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv7= nn.Conv2d(32,256, kernel_size=3, padding=1)
        self.conv8= nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.3)
        self.conv_drop = nn.Dropout2d(p=0.25)
        self.lin_drop = nn.Dropout(p=0.5)
        self.fc1=nn.Linear(256,1024)
        self.fc2=nn.Linear(1024,11)

    def forward(self,x):
        #1
        #print('level 0:{}'.format(x.shape)) #[N, 1, 128, 130]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        #print('level 2:{}'.format(x.shape)) #[N, 32, 128, 130]
        x = self.conv_drop(F.max_pool2d(x, 3))
        #print('level 2.5:{}'.format(x.shape)) #[N, 32, 42, 43]
        #2
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        #print('level 4:{}'.format(x.shape)) #[N, 64, 42, 43]
        x = self.conv_drop(F.max_pool2d(x, 3))
        #print('level 4.5:{}'.format(x.shape)) #[N, 64, 14, 14]
        #3
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        #print('level 6:{}'.format(x.shape)) #[N, 128, 14, 14]
        x = self.conv_drop(F.max_pool2d(x, 3))
        #print('level 6.5:{}'.format(x.shape)) #[N, 128, 4, 4]
        #4
        x = self.relu(self.conv7(x))
        #print('level 7:{}'.format(x.shape)) #[N, 256, 4, 4]
        x = self.relu(self.conv8(x))
        # global pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:]) 
        #print('level 8.5:{}'.format(x.shape)) #[N, 256, 1, 1]
        # print(x.shape)
        # reshape & fc. Torch infers this from other dimensions when one of the parameter is -1.
        x = x.view(-1, 256)
        #print('flatten:{}'.format(x.shape)) #[N, 256]
        x = self.lin_drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.softmax(x)
        return x
