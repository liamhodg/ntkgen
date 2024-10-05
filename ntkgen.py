import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from tqdm import tqdm
import zarr
import pickle
import time

import aioftp

from models import *

from functorch import make_functional_with_buffers, vmap, jacrev

def img_colorize(x):
    return x.repeat(3, 1, 1)

num_classes = 10

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

    def __init__(self, net, dataset, dtype, chkpath, nparams, device):
        self.dataset = dataset
        self.dtype = dtype
        self.num_classes = 10
        self.chkpath = chkpath

        self.net = net
        self.net.eval()
        self.fnet, self.params, self.buffers = make_functional_with_buffers(net)
        self.nparams = nparams
        self.device = device
    
    def fnet_single(self, params, buffers, x):
        return self.fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

    def get_jacobian(self, params, x):
        """Compute the Jacobian of the network for a minibatch x"""
        jac1 = vmap(jacrev(self.fnet_single), (None, None, 0))(params, self.buffers, x)
        jac1 = [j.detach().flatten(2).flatten(0,1) for j in jac1]
        return jac1
    
    def empirical_ntk(self, jac1, jac2):
        """From Jacobians J(x1) and J(x2), compute the empirical NTK block
        J(x1) @ J(x2).T / sqrt(d), where d is the number of model parameters. 
        
        The sqrt(d) normalization is performed to avoid overflow in the float16 case."""
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = (result/self.nparams**0.5).sum(0)
        return result
    
    def compute_ntk(self):
        """Compute the entire empirical NTK matrix. """
        # Get number of data points
        N = sum([x.shape[0] for (_, x) in self.dataset])
        total_num = N*self.num_classes
        filepath = '{}/ntk_{}'.format(self.chkpath, self.dtype)
        block_size = 5000
        #ntk = zarr.open(filepath+'.zarr', dtype=self.dtype, mode='w', 
        #                shape=(total_num,total_num),chunks=(block_size,block_size))
        if os.path.isfile(filepath+'.bin'):
            with open(filepath+'.txt', 'r') as f:
                idx_res = int(f.readlines()[0])
            ntk = np.memmap(filepath+'.bin', dtype=self.dtype, mode='r+', \
                        shape=(total_num,total_num))
        else:
            idx_res = 0
            ntk = np.memmap(filepath+'.bin', dtype=self.dtype, mode='w+', \
                            shape=(total_num,total_num))
        print('Start:',idx_res)
        nidx = 0
        nidy = 0
        num_iters = int(len(self.dataset)*(len(self.dataset)+1)/2)
        with tqdm(total=num_iters) as pbar:
            for idx, x1 in self.dataset:
                if idx < idx_res:
                    time.sleep(1e-1) # To avoid issues with progress bar
                    pbar.update(len(self.dataset)-idx)
                    nidx += x1.shape[0]*self.num_classes
                    continue
                x1d = x1.to(self.device)
                jac1 = self.get_jacobian(self.params, x1d)#.to(cpu)
                for idy, x2 in self.dataset:
                    if idy < idx:
                        nidy += x2.shape[0]*self.num_classes
                        continue
                    jac2 = self.get_jacobian(self.params, x2.to(self.device))#.to(cpu)
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
                with open(filepath+'.txt', 'w') as f:
                    f.write(str(idx))


class NTKGenerator(object):

    def __init__(self, name, chkpath, args):
        # Set device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print('Using CUDA')
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.name = name
        self.args = args
        self.net = self.get_net(name)
        self.nparams = self.count_parameters()
        print('number of parameters:', self.nparams)
        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.chkpath = chkpath
        self.acc = 0

        # Compute largest possible batch sizes for each GPU
        self.ntk_bs = []
        jac_size = self.nparams * self.num_classes * 8 * 3
        for idx in range(torch.cuda.device_count()):
            self.ntk_bs.append(int(torch.cuda.get_device_properties(idx).total_memory/jac_size))
        print('ntk batch size:', self.ntk_bs)
        if len(self.ntk_bs) == 1:
            self.ntk_bs = self.ntk_bs[0]

        # Load net if file exists
        self.load()

    def get_net(self, name):
        print('====================')
        print('==> Building model', name)
        width = self.args.width
        if name == 'resnet9':
            net = ResNet9(num_classes=num_classes, width=width)
        elif name == 'resnet18':
            net = ResNet18(num_classes=num_classes, width=width)
        elif name == 'resnet34':
            net = ResNet34(num_classes=num_classes, width=width)
        elif name == 'resnet50':
            net = ResNet50(num_classes=num_classes, width=width)
        elif name == 'resnet68':
            net = ResNet68(num_classes=num_classes, width=width)
        elif name == 'resnet101':
            net = ResNet101(num_classes=num_classes, width=width)
        elif name == 'resnet152':
            net = ResNet152(num_classes=num_classes, width=width)
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
        elif name == 'densenet121':
            net = DenseNet121()
        net = net.to(self.device)
        if self.device == 'cuda':
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
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.bs, shuffle=True)
        if os.path.isfile(self.chkpath+'/ntkset.pt'):
            self.ntkset = torch.load(self.chkpath+'/ntkset.pt')
        else:
            if self.args.subsample is not None:
                # Only subsample the dataset used for the NTK, not for training
                self.ntkset = torch.utils.data.Subset(self.ntkset, list(range(0, self.args.subsample)))
            self.ntkset = list(self.ntkset)
            torch.save(self.ntkset, self.chkpath + '/ntkset.pt')
        print('    sample size:', len(self.trainset))
        print('ntk sample size:', len(self.ntkset))
        self.ntkloader16 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs*4, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkloader32 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs*2, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkloader64 = torch.utils.data.DataLoader(self.ntkset, batch_size=self.ntk_bs, shuffle=False, pin_memory=True, num_workers=1)
        self.ntkset_f16 = [(idx, x.to(torch.float16)) for idx, (x,y) in enumerate(self.ntkloader16)]
        self.ntkset_f32 = [(idx, x.to(torch.float32)) for idx, (x,y) in enumerate(self.ntkloader32)]
        self.ntkset_f64 = [(idx, x.to(torch.float64)) for idx, (x,y) in enumerate(self.ntkloader64)]
    
    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
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
    
    def train(self, percent=99.5):
        """Train the network to >percent% accuracy. Percent
        defaults to 99.5."""
        if percent >= 100:
            percent = 100
        if self.acc < percent/100:
            for epoch in range(0, self.args.num_epochs):
                acc = self.train_epoch(epoch)
                if acc > percent/100:
                    self.save(acc)
                    return

    def save(self, acc):
        """Save the trained network to a checkpoint file on disk."""
        self.net.eval()
        state = {'net': self.net.state_dict(), 'acc':acc}
        torch.save(state, self.chkpath+'/net.pt')
        self.acc = acc

    def load(self):
        """Load the trained network from disk."""
        if not os.path.isfile(self.chkpath+'/net.pt'):
            return
        state = torch.load(self.chkpath+'/net.pt')
        self.net.load_state_dict(state['net'])
        self.net.eval()
        self.acc = state['acc']

    def compute_ntk(self, dtype):
        if dtype == 'float16':
            net = self.net.to(self.device).to(torch.float16)
            dataset = self.ntkset_f16
        elif dtype == 'float64':
            net = self.net.to(self.device).double()
            dataset = self.ntkset_f64
        else:
            net = self.net.to(self.device).to(torch.float32)
            dataset = self.ntkset_f32
        ntk = NTK(self.net, dataset, dtype, self.chkpath, self.nparams, self.device)
        ntk.compute_ntk()
