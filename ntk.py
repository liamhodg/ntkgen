import zarr
import numpy as np
import os
from tqdm import trange
from math import ceil

class NTK(object):

    def __init__(self, chkpath, dtype):
        self.dtype = dtype
        self.dataset = self.name.split('_')[0]
        self.chkpath = chkpath
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
            if not os.path.isdir(self.ntkpath + '.zarr'):
                raise Exception('No generated NTK to load')
            else:
                self.ntk = zarr.open(self.ntkpath+'.zarr', mode='r')
                self.is_memmap = False
                return
        self.ntk = np.memmap(self.ntkpath+'.bin', dtype=self.dtype, mode='r', \
                shape=(self.total_num,self.total_num))
        self.is_memmap = True
    
    def to_zarr(self):
        """Convert the NTK memmap into a zarr file for logdet computation
        and effective cold storage. Only the upper triangular part is kept,
        so filesize is reduced by a factor of roughly half."""
        if not self.is_memmap:
            raise Exception('NTK is already in .zarr format. No need to convert.')
        block_size = 5000
        num_blocks = ceil(self.total_num / block_size)
        ntk_zarr = zarr.open(self.ntkpath+'.zarr', dtype=self.dtype, mode='w', 
                       shape=(self.total_num,self.total_num),
                       chunks=(block_size,block_size))
        for idx in trange(num_blocks):
            i1 = idx*block_size
            i2 = (idx+1)*block_size
            ntk_zarr[i1:i2,i1:] = self.ntk[i1:i2,i1:]
        os.remove(self.ntkpath+'.bin')
    
    def __getitem__(self, key):
        return self.ntk[key]
    
    def __setitem__(self, key, value):
        # NTK is readonly
        pass
        
    def __repr__(self):
        return 'NTK([[{:.4f},{:.4f},{:.4f},...],...], N={})'.format(self.ntk[0,0],\
                    self.ntk[0,1],self.ntk[0,2],self.total_num)
    
    
