import argparse
import os
from ntkgen import NTKGenerator

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

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')

for name in args.models:
    for rep in range(args.repeats):
        # Get filename and pathname
        if args.subsample is None:
            subsample = 'F'
        else:
            subsample = args.subsample
        pathname = '{}_{}_{}_bs{}_w{}_s{}'.format(args.dataset,name,rep,args.bs, args.width, subsample)
        chkpath = 'ckpt/{}'.format(pathname)
        print('Model: {}'.format(pathname))

        # Check if the directory already exists
        if not os.path.exists(chkpath):
            os.makedirs(chkpath)
            
        # Prepare the dataset for training
        gen = NTKGenerator(name, chkpath, args)
        gen.prepare_dataset(args.dataset)
        gen.train(99.5)
        gen.compute_ntk('float16')
        gen.compute_ntk('float32')
        # WARNING: Computing NTK in double precision is EXTREMELY SLOW!! Consider commenting
        #          following line to avoid long computation times.
        # gen.compute_ntk('float64')
        
