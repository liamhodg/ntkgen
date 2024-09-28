'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import copy
import math
from functorch import make_functional_with_buffers, vmap, jacrev
from tqdm import tqdm, trange
import zarr
import pickle
import os
import argparse

from models import *


parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--models', nargs='+',type=str, help='models', required=True)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset',default='cifar',type=str,help='dataset (cifar or imagenet)')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--width',default=64, type=int, help='model width')
parser.add_argument('--num_epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--subsample', default=None, type=int, help='number in subsampled dataset')
parser.add_argument('--repeats',default=1, type=int, help='number of times to train the model')
parser.add_argument('--init',action="store_true",help='use initialized model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
    print('Using CUDA')
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = args.num_epochs

def img_colorize(x):
    return x.repeat(3, 1, 1)

num_classes = 10

# Data
print('==> Preparing data..')

transform_train = {
    'cifar': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'mnist': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.Resize((32,32)),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Lambda(img_colorize)]),
    'svhn': transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]),
    
}

transform_test = {
    'cifar': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'mnist': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(img_colorize)
    ]),
    'svhn': transforms.Compose([
        transforms.ToTensor()]),
}

class NTK(object):

    def __init__(self, net, dataset, dtype, chkpath, nparams):
        self.net = net
        self.dataset = dataset
        self.dtype = dtype
        self.num_classes = 10
        self.chkpath = chkpath
        self.net.eval()
        self.fnet, self.params, self.buffers = make_functional_with_buffers(net)
        self.nparams = nparams
    
    def fnet_single(self, params, buffers, x):
        return self.fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

    def get_jacobian(self, fnet_single, params, x):
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, None, 0))(params, self.buffers, x)
        jac1 = [j.detach().flatten(2).flatten(0,1) for j in jac1]
        return jac1

    def _to_device_array(self, x, dev):
        x = [xi.to(dev) for xi in x]
        return x
    
    def empirical_ntk(self, jac1, jac2):
        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = (result/self.nparams**0.5).sum(0)
        return result
    
    def compute_ntk(self):
        # Get number of data points
        N = 0
        dataset = self.dataset
        N = sum([x.shape[0] for (idx, x) in dataset])
        print(N)
        total_num = N*self.num_classes
        block_size = 5000
        big = total_num > 10*block_size
        if big:
            ntk = zarr.open(self.chkpath[:-4]+'_'+self.dtype+'_ntk.zarr', dtype=self.dtype, mode='w', 
                            shape=(total_num,total_num),chunks=(block_size,block_size))
        else:
            ntk = np.memmap(self.chkpath[:-4]+'_'+self.dtype+'_ntk.bin', dtype=self.dtype, mode='w+', 
                            shape=(total_num,total_num))
        nidx = 0
        nidy = 0
        with tqdm(total=int(len(dataset)*(len(dataset)+1)/2)) as pbar:
            for idx, x1 in dataset:
                x1d = x1.to(device)
                jac1 = self.get_jacobian(self.fnet_single, self.params, x1d)#.to(cpu)
                for idy, x2 in dataset:
                    if idy < idx:
                        nidy += x2.shape[0]*self.num_classes
                        continue
                    jac2 = self.get_jacobian(self.fnet_single, self.params, x2.to(device))#.to(cpu)
                    J = self.empirical_ntk(jac1, jac2).to('cpu')
                    del jac2
                    incx = J.shape[0]
                    incy = J.shape[1]
                    ntk[nidx:nidx+incx,nidy:nidy+incy] = J
                    if idx != idy:
                        ntk[nidy:nidy+incy,nidx:nidx+incx] = J.T
                    nidy += incy
                    pbar.update(1)
                nidx += incx
                nidy = 0
        #return ntk

class NTKGenerator(object):

    def __init__(self, name, chkpath):
        self.name = name
        self.net = self.get_net(name)
        self.nparams = self.count_parameters()
        print('number of parameters:', self.nparams)
        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.chkpath = chkpath
        self.ntk_bs = int(9*1024*1024*1024/(self.nparams*self.num_classes*8*3))
        print('ntk batch size:', self.ntk_bs)

    def get_net(self, name):
        print('====================')
        print('==> Building model', name)
        if name == 'resnet9':
            net = ResNet9(num_classes=num_classes, width=args.width)
        elif name == 'resnet18':
            net = ResNet18(num_classes=num_classes, width=args.width)
        elif name == 'resnet34':
            net = ResNet34(num_classes=num_classes, width=args.width)
        elif name == 'resnet50':
            net = ResNet50(num_classes=num_classes, width=args.width)
        elif name == 'resnet68':
            net = ResNet68(num_classes=num_classes, width=args.width)
        elif name == 'resnet101':
            net = ResNet101(num_classes=num_classes, width=args.width)
        elif name == 'resnet152':
            net = ResNet152(num_classes=num_classes, width=args.width)
        elif name == 'mobilenet':
            net = MobileNet(num_classes=num_classes)
        elif name == 'mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        elif name == 'vgg11':
            net = VGG('VGG11')
        elif name == 'vgg13':
            net = VGG('VGG13')
        elif name == 'lenet':
            net = LeNet()
        elif name == 'wrn-28-2':
            net = WRN(depth=28, widening_factor=2, num_classes = num_classes)
        elif name == 'wrn-28-5':
            net = WRN(depth=28, widening_factor=5, num_classes = num_classes)
        elif name == 'wrn-28-10':
            net = WRN(depth=28, widening_factor=10, num_classes = num_classes)
        elif name == 'logistic':
            net = Logistic()
        elif name == 'twolayer':
            net = TwoLayer()
        elif name == 'twolayermini':
            net = TwoLayerBottle()
        elif name == 'densenet121':
            net = DenseNet121()
        net = net.to(device)
        if device == 'cuda':
            cudnn.benchmark = True
        return net

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def prepare_dataset(self, dataset):
        if dataset == 'cifar':
            self.trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=True, 
                                                    transform=transform_train['cifar'])
            self.ntkset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=True, 
                                                     transform=transform_test['cifar'])
        elif dataset == 'mnist':
            self.trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, 
                                                  transform=transform_train['mnist'])
            self.ntkset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, 
                                                   transform=transform_test['mnist'])
        elif dataset == 'kmnist':
            self.trainset = torchvision.datasets.KMNIST(root='./kmnist_data', train=True, download=True, 
                                                   transform=transform_train['mnist'])
            self.ntkset = torchvision.datasets.KMNIST(root='./kmnist_data', train=True, download=True,
                                                   transform=transform_test['mnist'])
        elif dataset == 'fmnist':
            self.trainset = torchvision.datasets.FashionMNIST(root='./fmnist_data', train=True, download=True, 
                                                   transform=transform_train['mnist'])
            self.ntkset = torchvision.datasets.FashionMNIST(root='./fmnist_data', train=True, download=True,
                                                   transform=transform_test['mnist'])
        elif dataset == 'svhn':
            self.trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train', download=True, 
                                                 transform=transform_train['svhn'])
            self.ntkset = torchvision.datasets.SVHN(root='./svhn_data', split='train', download=True, 
                                                  transform=transform_test['svhn'])
        self.ntkset = torch.utils.data.Subset(self.ntkset, list(range(0, args.subsample)))
        self.ntkset = list(self.ntkset)
        print('ntkset', len(self.ntkset))
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=args.bs, shuffle=True)
        self.ntkloader16 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs*4, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkloader32 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs*2, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkloader64 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkset_f16 = [(idx, x.to(torch.float16)) for idx, (x,y) in enumerate(self.ntkloader16)]
        self.ntkset_f32 = [(idx, x.to(torch.float32)) for idx, (x,y) in enumerate(self.ntkloader32)]
        self.ntkset_f64 = [(idx, x.to(torch.float64)) for idx, (x,y) in enumerate(self.ntkloader64)]
    
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx % 10) == 0:
                print(self.name,'|',batch_idx,'of', len(self.trainloader), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.scheduler.step()
        return correct/total

    def save(self):
        self.net.eval()
        state = {'net': self.net.state_dict()}
        torch.save(state, self.chkpath)

    def compute_ntk(self, dtype):
        if dtype == 'float16':
            net = self.net.to(device).to(torch.float16)
            dataset = self.ntkset_f16
        elif dtype == 'float64':
            net = self.net.to(device).double()
            dataset = self.ntkset_f64
        else:
            net = self.net.to(device).to(torch.float32)
            dataset = self.ntkset_f32
        ntk = NTK(self.net, dataset, dtype, self.chkpath, self.nparams)
        ntk.compute_ntk()

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
for name in args.models:
    for rep in range(args.repeats):
        print(name, rep, args.bs, args.width)
        pathname = '{}_{}_{}_bs{}_w{}'.format(args.dataset,name,rep,args.bs, args.width)
        chkpath = 'ckpt/{}.pth'.format(pathname)
        if os.path.isfile(chkpath):
            continue
        start_epoch = 0
        comp = NTKGenerator(name, chkpath)
        comp.prepare_dataset(args.dataset)
        for epoch in range(start_epoch, args.num_epochs):
            acc = comp.train(epoch)
            if acc > 0.995:
                comp.save()
                break
        comp.compute_ntk('float16')
        comp.compute_ntk('float32')
        comp.compute_ntk('float64')
        
        