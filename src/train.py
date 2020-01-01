import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
from visdom import Visdom
from dataloader import Style_Dataset
import datetime
from model import *
#from fcn import *    ####fcn model
from utils import *
from sklearn.manifold import TSNE
from matplotlib import cm

vali_split = .1
config={
    # path
    'plot_recon_test':'../plots/recon_fcn_5&3/test',  ###change dir
    'plot_recon_train':'../plots/recon_fcn_5&3/train',
    'checkpt':'../checkpoint/1s_ver1_z128_ker53(2s_data)',
    'tsne':'../plots/tsne/encoder/test_4000_lastlayer/'  ##change test vali name
}

def plot_with_labels(lowDWeights, labels, epoch, test):
    #plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    print(lowDWeights.shape)
    plt.scatter(X, Y, c=labels, s=10, cmap=plt.cm.get_cmap("jet", 8))
    '''
    for x, y, s in zip(X, Y, labels):
        #print(labels)
        c = cm.rainbow(int(255 * s / 7));
        plt.text(x, y, s, backgroundcolor=c, fontsize=2)
    '''
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.colorbar(ticks=range(8))
    plt.title('Visualize Classification')
    print('Generate cluster img')
    plt.savefig(config['tsne']+'%03d.png'%epoch, format='png')
    plt.close()
    '''
    if test:
        plt.savefig('../plots/tsne/test/%03d.png'%epoch, format='png')
    else:
        plt.savefig('../plots/tsne/train/%03d.png'%epoch, format='png')
    '''

def to_img(x):
    x=x.cpu().data.numpy() 
    #print('in')
    #print(x.shape)#(16,1,128,44)
    x=x.reshape([-1,128,44])
    #print(x.shape)#(16,128,44)

    return x

def plot_reconstruction(device, model, loader, epoch, train=False, save=True):
    model.eval()
    data, _ = next(iter(loader))
    #print(data.shape)
    data = data.to(device)
    _, decoded_imgs = model(data)
    true_imgs = data
    
    true_imgs=to_img(true_imgs)
    decoded_imgs=to_img(decoded_imgs)
    n=16
    plt.figure(figsize=(7*4,4*4))
    #plt.clf()
    for i in range(n):
        #display original
        plt.subplot(4, n/2, i + 1)
        #print(true_imgs[i].shape)
        librosa.display.specshow(true_imgs[i], hop_length=512)
        if(i==n-1):
            plt.colorbar()

        #display reconstruction
        plt.subplot(4, n/2, i + 1 + n)
        librosa.display.specshow(decoded_imgs[i], hop_length=512)
        if(i==n-1):
            plt.colorbar()

    if save:
        if train:
            plt.savefig(config['plot_recon_train']+'/%03d.png'%epoch, format='png')
        else:
            plt.savefig(config['plot_recon_test']+'/%03d.png'%epoch, format='png')
    plt.close('all')


def train(args, model, device, data_loader, optimizer, c_criterion, re_criterion, epoch):
#def train(args, c_model, ae_model, device, data_loader, c_optimizer, re_optimizer, c_criterion, re_criterion, epoch):
    model.train()
    #c_model.train()
    #ae_model.train()
    c_epoch_loss = 0
    #re_epoch_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        out_c,_ = model(data)
        #out_c, out_re = model(data)

        c_loss = c_criterion(out_c, target)
        #re_loss = re_criterion(out_re,data)
        loss = c_loss
        #loss = c_loss + re_loss
        c_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        '''
        c_optimizer.zero_grad()
        re_optimizer.zero_grad()
        out_c=c_model(data)
        out_re=ae_model(data)
        c_loss = c_criterion(out_c, target)
        re_loss = re_criterion(out_re,data)
        c_loss.backward()
        c_optimizer.step()
        re_loss.backward()
        re_optimizer.step()
        c_epoch_loss += c_loss.item()
        re_epoch_loss += re_loss.item()
        '''
        #accurency
        pred = out_c.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), c_loss.item()))
        # loss.item()
    c_epoch_loss/=(batch_idx+1)
    #re_epoch_loss/=(batch_idx+1)
    acc = 100. *correct/ len(data_loader.dataset)
    print('Train Epoch: {}, Accurency: {}/{} ({:.0f}%), Loss:{}'.format(epoch, 
        correct, len(data_loader.dataset), 
        acc, c_epoch_loss))
    return c_epoch_loss, acc

def test(args, model, device, data_loader, c_criterion, re_criterion, epoch, test=False):
#def test(args, c_model, ae_model, device, data_loader, c_criterion, re_criterion, epoch):
    #c_model.eval()
    #ae_model.eval()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            #out_c, _ = model(data)
            out_c,last_layer = model(data)
            loss = c_criterion(out_c, target)
            #print(out_c.shape)
            '''
            out_c, out_re = model(data)
            c_loss = c_criterion(out_c, target)
            re_loss = re_criterion(out_re,data)
            loss = c_loss + re_loss
            '''
            test_loss += loss.item()
            
            '''
            out_c=c_model(data)
            #out_re=ae_model(data)
            c_loss = c_criterion(out_c, target)
            #re_loss = re_criterion(out_re,data)
            test_loss += c_loss.item()
            '''
            #accurency
            pred = out_c.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            '''
            #visualization
            if test:
                if batch_idx==0:
                    #all_embs=out_c.cpu().data.numpy()
                    all_embs=last_layer.cpu().data.numpy()
                    all_labels=target.cpu().numpy()
                else:
                    #all_embs=np.concatenate((all_embs,out_c.cpu().data.numpy()), axis=0)
                    all_embs=np.concatenate((all_embs,last_layer.cpu().data.numpy()), axis=0)
                    all_labels=np.concatenate((all_labels,target.cpu().numpy()), axis=0)
            
        tsne =TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)  
        all_embs = tsne.fit_transform(all_embs) 
        plot_with_labels(all_embs, all_labels, epoch, test)
        '''
            

    test_loss/=(batch_idx+1)
    acc = 100. *correct/ len(data_loader.dataset)
    print('Test Epoch: {}, Accurency: {}/{} ({:.0f}%), Loss:{}'.format(epoch, 
        correct, len(data_loader.dataset), acc, test_loss))

    return test_loss, acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Music Style')
    parser.add_argument('--checkpoint', type=str, help="checkpoint of pre-trained model")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')   ########### 128
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--log-interval', type=int, default=130, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")    
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # transform
    train_mean, train_std=compute_mean_std('data_1s')
    img_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(train_mean,),std=(train_std,))
    ])
    
    # train & validation dataset
    train_data=Style_Dataset('npy/train/data_2s/','data.npy','label.npy', transform = img_transform)
    #train_data=Style_Dataset('train/data_1s/','data.npy','label.npy')
    train_size = int((1-vali_split)* len(train_data))
    vali_size = len(train_data) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(train_data, [train_size, vali_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=vali_dataset, batch_size=args.batch_size, shuffle=False)
    
    # test dataset
    test_dataset = Style_Dataset('npy/test/data_2s/','data.npy','label.npy', transform = img_transform)
    #test_dataset = Style_Dataset('test/data_1s/','data.npy','label.npy')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    '''
    ##### cluster dataset
    test_dataset = Style_Dataset('cluster_npy/test/data_1s/','data.npy','label.npy', transform = img_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=400, shuffle=False)
    '''
    

    # Setup model 
    model=Encoder().cuda()
    '''
    encoder = Encoder()
    decoder = Decoder()
    model = Autoencoder(encoder, decoder).cuda()
    '''
    '''
    encoder = Encoder()
    decoder = Decoder()
    classifier = Classifier(encoder).cuda()
    auto = Autoencoder(encoder, decoder).cuda()
    '''
    #model=Encoder().cuda()
    classifier_criterion = torch.nn.CrossEntropyLoss() #elementwise_mean
    autoencoder_criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    c_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #re_optimizer = torch.optim.Adam(auto.parameters(), lr=args.lr)
    summary(model,(1,128,44))

    
    # resume
    save_dict={}
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        c_optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch']+1
    else:
        start_epoch=1
    
    # make directory
    mkdir(config['checkpt'])
    #mkdir(config['plot_recon_train'])
    #mkdir(config['plot_recon_test'])
    mkdir(config['tsne'])

    # draw 
    vis = Visdom(env='music')
    #### python -m visdom.server / http://localhost:8097

    print('Start training...')
    starttime = datetime.datetime.now()
    # Start training
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, c_optimizer, classifier_criterion, autoencoder_criterion, epoch)
        _, test_acc= test(args, model, device, test_loader, classifier_criterion, autoencoder_criterion, epoch, test=True)
        
        if (epoch%3==0):
            _, vali_acc= test(args, model, device, validation_loader, classifier_criterion, autoencoder_criterion, epoch, test=False)
            vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([vali_acc]), win='Vali_Acc', update='append' if epoch+1 >0  else None,
            opts={'title': '1s_Vali Acc'})
            #plot_reconstruction(device, model, validation_loader, epoch, train=True)
            #plot_reconstruction(device, model, test_loader, epoch, train=False)
        
        #train_loss, train_acc = train(args, classifier, auto, device, train_loader, c_optimizer, re_optimizer, classifier_criterion, autoencoder_criterion, epoch)
        #vali_loss, vali_acc= test(args, classifier, auto, device, validation_loader, classifier_criterion, autoencoder_criterion, epoch)
        #test_loss, test_acc= test(args, classifier, auto, device, test_loader, classifier_criterion, autoencoder_criterion, epoch)
        
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([train_loss]), win='Train_loss', update='append' if epoch+1 >0  else None,
            opts={'title': '1s_Train loss'})
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([train_acc]), win='Train_Acc', update='append' if epoch+1 >0  else None,
            opts={'title': '1s_Train Acc'})
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([test_acc]), win='Test_Acc', update='append' if epoch+1 >0  else None,
            opts={'title': '1s_Test_Acc'})

        
        if(epoch%5==0):
            save_dict['model']=model.state_dict()
            save_dict['epoch']=epoch
            save_dict['opt']=c_optimizer.state_dict()
            torch.save(save_dict,config['checkpt']+'/style_%03d.pt'%(epoch))

        if(epoch%20==0):
            endtime = datetime.datetime.now()
            print (endtime - starttime)
        
        

    

if __name__ == '__main__':
    main()