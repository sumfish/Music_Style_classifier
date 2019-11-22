import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
from visdom import Visdom
from dataloader import Style_Dataset
from model import S_Model

vali_split = .2

def train(args, model, device, data_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        #print(target)
        
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
        # loss.item()
    return epoch_loss

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Music Style')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')   ########### 128
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    '''
    # train & validation
    train_data=Style_Dataset('data.npy','label.npy')
    train_size = int((1-vali_split)* len(train_data))
    vali_size = len(train_data) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(train_data, [train_size, vali_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=vali_dataset, batch_size=args.batch_size, shuffle=True)
    '''
    # test mnist
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs) 
    '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    '''
    # Setup model 
    model=S_Model().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # draw 
    vis = Visdom(env='music')
    #### python -m visdom.server / http://localhost:8097
    print('Start training...')
    # Start training
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, criterion, epoch)
        #test(args, model, device, test_loader)
        vis.line(X=torch.FloatTensor([epoch+1]), Y=torch.FloatTensor([train_loss]), win='train_loss', update='append' if epoch+1 >0  else None,
            opts={'title': 'train loss'})
        

if __name__ == '__main__':
    main()