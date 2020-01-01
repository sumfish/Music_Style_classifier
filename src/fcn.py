import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
zsize=128
ch = 64
num_classes =8

# frame-level paper
class block(nn.Module):
    #def __init__(self, inp, out):
    def __init__(self, inp, out, kernel):
        super(block, self).__init__()
        if kernel==3:
            last_kernel=1
        else: 
            last_kernel=5
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (kernel,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (kernel,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)
        self.up = nn.Conv2d(inp, out, (last_kernel,1), padding=(0,0))
    
    def forward(self, x):
        #print('block x:{}'.format(x.shape))  #shape(N,C,128,87)
        out = self.conv1(self.bn1(x)) #before is a cnn layer
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        #print('f(x):{}'.format(out.shape))
        out += self.up(x) ##########################
        #print('block x:{}'.format(out.shape))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(ch*3, zsize)
        self.fc2 = nn.Linear(zsize, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            nn.Conv2d(1, ch, (5,1), padding=(1,0)),
            block(ch, ch*2, 5),
            #nn.Conv2d(1, ch, (3,1), padding=(1,0)),
            #block(ch, ch*2),
            nn.Dropout(p=0.25),
            nn.MaxPool2d((3,1),(3,1)), #(42,87)
            
            #block(ch*2, ch*3),
            block(ch*2, ch*3, 3),
            nn.BatchNorm2d(ch*3),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d((3,1),(3,1)),
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
        '''
        x = self.avgpool(x)
        #print('level 2:{}'.format(x.shape))
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        x = self.lin_drop(F.relu(self.fc1(x)))
        #print('level 4:{}'.format(x.shape))
        '''
        #x = F.softmax(self.fc2(x), dim=0) ####classifier
        #x = self.fc2(x) 
        #print('level 5:{}'.format(x.shape))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        ch=64
        self.dfc2 = nn.Linear(zsize, 512)
        self.dfc1 = nn.Linear(512,32*10*3)
        #self.dfc1 = nn.Linear(512,ch*6*6) #64
        #self.upsample1=nn.Upsample(size=(128,44),mode='bilinear') #(44)
        #self.upsample2=nn.Upsample(size=(36,24),mode='bilinear') #(12,12)
        self.upsample3=nn.Upsample(scale_factor=3)
        #self.upsample3=nn.Upsample(size=(30,11))
        self.upsample2=nn.Upsample(scale_factor=2)
        self.upsample1=nn.Upsample(scale_factor=2)
        #self.dconv3 = nn.ConvTranspose2d(ch, ch*4, (3,1), stride = 2, padding = (1,0)) #192
        #self.dconv2 = nn.ConvTranspose2d(ch*4, ch, (3,1), stride = (3,1), padding = (2,0))
        #self.dconv1 = nn.ConvTranspose2d(ch, 1, (3,1), stride = 1, padding = (4,1))
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.bn3 = nn.BatchNorm2d(ch*4)
        self.conv_drop = nn.Dropout2d(p=0.2)
        #self.con3=nn.Conv2d(ch, ch*4, (3,1), padding=(2,0))
        #self.con2=nn.Conv2d(ch*4, ch, (3,1), padding=(1,0))
        #self.con1=nn.Conv2d(ch, 1, (3,1), padding=(1,0))
        self.con3=nn.Conv2d(ch, ch*4, 3, padding=(1,1))
        self.con2=nn.Conv2d(ch*4, ch, 3, padding=(0,0))
        self.con1=nn.Conv2d(ch, 1, (3,1), padding=(1,0)) ##################
        self.head = nn.Sequential(
            #nn.BatchNorm2d(inp), ###############
            #nn.Upsample(scale_factor=3),
            #nn.Conv2d(32, ch*3, 3, padding=(2,2)),
            #nn.Upsample(scale_factor=2),
            block(ch*3, ch*2, 3),
            nn.Dropout(p=0.25),
            #nn.MaxUnpool2d((3,1),(3,1)), #(42,87)
            nn.Upsample(size=(132,44)),
            block(ch*2, ch, 5),
            #nn.Dropout(p=0.3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            
            #nn.MaxPool2d((3,1),(3,1)),
            #nn.Conv2d(fre*3, fre*2, (3,1), padding=(1,0))
        )

    
    def forward(self, x):
        '''
        x = F.relu(self.dfc2(x))
        #print('de level 1:{}'.format(x.shape))
        x = F.relu(self.dfc1(x))
        #print('de level 2:{}'.format(x.size()))
        x = x.view(x.size(0),32,10,3)
        #print('de level 2.5:{}'.format(x.size()))
        '''
        #print('de original:{}'.format(x.shape))
        x = self.head(x)
        #print('de level 1:{}'.format(x.shape))
        x = self.con1(x)
        #print('de level 2:{}'.format(x.size()))
        #input()
        '''
        x = F.relu(self.dfc2(x))
        #print('de level 1:{}'.format(x.shape))
        x = F.relu(self.dfc1(x))
        #print('de level 2:{}'.format(x.size()))
        x = x.view(x.size(0),64,11,4)
        #print('de level 2.5:{}'.format(x.size()))
        x=self.upsample3(x)
        #print('de level 3:{}'.format(x.size()))
        #x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn3(self.con3(x)))
        #print('de level 5:{}'.format(x.size()))
        x=self.upsample2(x)
        #print('de level 6:{}'.format(x.size()))
        #x = F.relu(self.bn2(self.dconv2(x)))  
        x = F.relu(self.bn2(self.con2(x)))
        #print('de level 7:{}'.format(x.size()))
        #x = self.dconv1(x)
        x=self.upsample1(x)
        x = self.bn1(self.con1(x))  ############################bn??????
        #x = self.con1(x)
        #print('de level 8:{}'.format(x.size()))
        #input()
        #x = F.sigmoid(x)
        '''
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super(Autoencoder,self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_layer = nn.Sequential(
            nn.Linear(ch*3, zsize),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(zsize,num_classes),
            )
    
    def forward(self,x):
        #print(x.shape)
        en_out = self.encoder(x)
        x=en_out
        x_re = self.decoder(x)
        #print(x_re.shape)
        # classifier by fc
        x = self.pool(en_out)
        #x = x.view(-1, 192)
        x = torch.flatten(x, 1)
        x = self.classifier_layer(x)
        x_c = F.softmax(x, dim=0)
        return x_c, x_re
        #return x_re

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