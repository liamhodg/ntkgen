import zarr
import numpy as np
import os
from tqdm import trange

class NTK(object):

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype
        self.dataset = self.name.split('_')[0]
        self.chkpath = 'ckpt/' + self.name
        self.ntkpath = '{}/ntk_{}'.format(self.chkpath, self.dtype)
        if self.dataset == 'cifar':
            self.total_num = 50000*10
        elif self.dataset == 'svhn':
            self.total_num = 73257*10
        else:
            self.total_num = 60000*10
        self.ntk_load()

    def ntk_load(self):
        """Load the memmap'ed NTK matrix with given name and datatype"""
        if not os.path.isfile(self.ntkpath + '.bin'):
            if not os.path.isfile(self.ntkpath + '.zarr'):
                raise Exception('No generated NTK to load')
            else:
                self.ntk = zarr.open(self.ntkpath+'.zarr', mode='r')
                return
        self.ntk = np.memmap(self.ntkpath, dtype=self.dtype, mode='r', \
                shape=(self.total_num,self.total_num))
    
    def to_zarr(self):
        block_size = 5000
        ntk_zarr = zarr.open(self.ntkpath+'.zarr', dtype=self.dtype, mode='w', 
                       shape=(self.total_num,self.total_num),
                       chunks=(block_size,block_size))
        for row in trange(self.total_num):
            ntk_zarr[row,row:] = self.ntk[row,row:]
    