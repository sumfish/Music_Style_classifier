import torch.nn as nn
import torch
import torch.nn.functional as F
zsize=128
num_classes =11

## each cnn is zero-padded
class S2016_Model(torch.nn.Module):
    def __init__(self):
        super(S2016_Model,self).__init__()
        self.conv1= nn.Conv2d(1,32, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(32,32, kernel_size=3, padding=1)
        self.conv3= nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.conv4= nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.conv5= nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.conv6= nn.Conv2d(128,128, kernel_size=3, padding=1)
        self.conv7= nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.conv8= nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.3)
        self.conv_drop = nn.Dropout2d(p=0.25)
        self.lin_drop = nn.Dropout(p=0.5)
        self.fc1=nn.Linear(64,256)
        self.fc2=nn.Linear(256,8)
        #self.fc1=nn.Linear(256,512)
        #self.fc2=nn.Linear(512,8)

    def forward(self,x):
        #1
        #print('level 0:{}'.format(x.shape)) #[N, 1, 128, 130]
        x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))
        #print('level 2:{}'.format(x.shape)) #[N, 32, 128, 130]
        x = self.conv_drop(F.max_pool2d(x, 3))
        #x = F.max_pool2d(x, 3)
        #print('level 2.5:{}'.format(x.shape)) #[N, 32, 42, 43]
        #2
        #x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        #print('level 4:{}'.format(x.shape)) #[N, 64, 42, 43]
        #x = self.conv_drop(F.max_pool2d(x, 3))
        #x = F.max_pool2d(x, 3)
        #print('level 4.5:{}'.format(x.shape)) #[N, 64, 14, 14]
        #3
        #x = self.relu(self.conv5(x))
        #x = self.relu(self.conv6(x))
        #print('level 6:{}'.format(x.shape)) #[N, 128, 14, 14]
        # = self.conv_drop(F.max_pool2d(x, 3))
        #print('level 6.5:{}'.format(x.shape)) #[N, 128, 4, 4]
        #4
        x = self.relu(self.conv7(x))
        #print('level 7:{}'.format(x.shape)) #[N, 256, 4, 4]
        #x = self.relu(self.conv8(x))
        # global pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        #print('level 8.5:{}'.format(x.shape)) #[N, 256, 1, 1]
        #print(x.shape)
        # reshape & fc. Torch infers this from other dimensions when one of the parameter is -1.
        #x = x.view(-1, 256)
        x = x.view(-1, 64)
        #print('flatten:{}'.format(x.shape)) #[N, 256]
        x = self.lin_drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        #print(x)
        x = F.softmax(x, dim=0)
        #print(x)
        return x

# frame-level paper
class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)
        self.up = nn.Conv2d(inp, out, (1,1), padding=(0,0))
    
    def forward(self, x):
        #print('block x:{}'.format(x.shape))  #shape(N,C,128,87)
        out = self.conv1(self.bn1(x)) #before is a cnn layer
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        #print('f(x):{}'.format(out.shape))
        out += self.up(x) ##########################
        #print('block x:{}'.format(self.up(x).shape))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        fre = 64
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(fre*3, zsize)
        self.fc2 = nn.Linear(zsize, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            nn.Conv2d(1, fre, (3,1), padding=(1,0)),
            block(fre, fre*2),
            nn.Dropout(p=0.25),
            nn.MaxPool2d((3,1),(3,1)), #(42,87)
            
            block(fre*2, fre*3),
            #nn.Dropout(p=0.3),
            nn.BatchNorm2d(fre*3),
            nn.ReLU(inplace=True),
            #nn.Conv2d(fre*3, fre*2, (3,1), padding=(1,0))
        )
        

    def forward(self, _input):
        '''
        original:torch.Size([16, 1, 128, 87])
        level 1:torch.Size([16, 512, 14, 87])
        level 2:torch.Size([16, 512, 1, 1])
        level 3:torch.Size([16, 512])
        level 4:torch.Size([16, 11])
        level 5:torch.Size([16, 11])
        '''
        x = _input
        #print('original:{}'.format(x.shape))
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = self.avgpool(x)
        #print('level 2:{}'.format(x.shape))
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        x = self.lin_drop(F.relu(self.fc1(x)))
        #print('level 4:{}'.format(x.shape))
        x = F.softmax(self.fc2(x), dim=0) ####classifier
        #x = self.fc2(x) 
        #print('level 5:{}'.format(x.shape))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dfc2 = nn.Linear(zsize, 2048)
        #self.dfc1 = nn.Linear(2048,128*6*6)
        self.dfc1 = nn.Linear(zsize,128*6*6)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv3 = nn.ConvTranspose2d(128, 192, (3,1), stride = 2, padding = (1,0))
        self.dconv2 = nn.ConvTranspose2d(192, 64, (3,1), stride = (3,1), padding = (2,0))
        self.dconv1 = nn.ConvTranspose2d(64, 1, (3,1), stride = 1, padding = (4,1))

    def forward(self, x):
        #x = F.relu(self.dfc2(x))
        #('de level 1:{}'.format(x.shape))
        x = F.relu(self.dfc1(x))
        #print('de level 2:{}'.format(x.size()))
        x = x.view(x.size(0),128,6,6)
        #print('de level 2.5:{}'.format(x.size()))
        x=self.upsample1(x)
        #print('de level 3:{}'.format(x.size()))
        x = F.relu(self.dconv3(x))
        #print('de level 5:{}'.format(x.size()))
        x=self.upsample1(x)
        #print('de level 6:{}'.format(x.size()))
        x = self.dconv2(x)  
        #print('de level 7:{}'.format(x.size()))
        x = F.relu(x)
        #print('de level 8:{}'.format(x.size()))
        x = self.dconv1(x)
        #print('de level 9:{}'.format(x.size()))
        #x = F.sigmoid(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super(Autoencoder,self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.classifier = nn.Sequential(
            #nn.Linear(zsize,64),
            #nn.ReLU(inplace=True),
            #nn.Linear(64,num_classes),
            nn.Linear(zsize,num_classes),
            #nn.ReLU(inplace=True),
            )
    
    def forward(self,x):
        x = self.encoder(x)
        x_re = self.decoder(x)

        #x = self.classifier(x)
        #x_c = F.softmax(x, dim=0)
        #return x_c, x_re
        return x_re

class Classifier(nn.Module):
    def __init__(self, enc):
        super(Classifier,self).__init__()
        self.encoder = enc
        self.classifier = nn.Sequential(
            #nn.Linear(zsize,64),
            #nn.ReLU(inplace=True),
            #nn.Linear(64,num_classes),
            nn.Linear(zsize,num_classes),
            #nn.ReLU(inplace=True),
            )
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.classifier(x)
        x_c = F.softmax(x, dim=0)
        return x_c

'''
de=Decoder()
print(de)


en=Encoder()
print(en)
'''