#! /usr/bin/env python

# -------
# Install
# -------
#
# memdet is available on detkit version 0.5.4 or higher. Make sure to install
# the latest version:
#
#     pip install --upgrade detkit
#
# ------
# Output
# ------
#
# This script writes a file with .npz suffix in the same directory as this
# script's location. To read the output:
#
#     import numpy
#     data = numpy.load('output_file.npz', allow_pickle=True)
#
#     # diagonal elements of the U matrix in LU decomposition
#     diag = data['diag']
#
#     # information about computation
#     info = data['info'].item()
#
#     # Print info dictionary
#     from pprint import pprint
#     pprint(info)


# =======
# Imports
# =======

import numpy
import zarr
from pprint import pprint
import detkit
import os
import psutil
import tqdm
from ntk import NTK

def block_diag(A, num_classes=10):
    n = A.shape[0] // num_classes
    m = num_classes
    dimat = numpy.zeros(A.shape, dtype='float64')
    for i in range(n):
        dimat[i*m:(i+1)*m,i*m:(i+1)*m] = A[i*m:(i+1)*m,i*m:(i+1)*m]
    return dimat


# ====
# main
# ====

def main():

    dirs = [f.path for f in os.scandir('ckpt') if f.is_dir()]

    #for dir in dirs:

    # base_filename: str
    # The base filename of the input file, without '.zarr' suffix. This name
    # together with '.npz' file extension will also be used for the output file
    # name.
    dtypes = ['float64']
    # scratch_dir: str or None
    # The directory where memdet will create temporary scratch file. If None,
    # the default OS's tmp directorty will be used. In Linux, this is '/tmp'.
    # This directory should have an available space of at least the size of the
    # input matrix.
    # Note: Often '/tmp' directory is not assigned a large space. Make sure
    # '/tmp' has enough space, otherwise, specify another directory other than
    # leaving this argument as None.
    scratch_dir = '/mnt/Rocket/tmp'

    # -----------
    # Computation
    # -----------
    for chkpath in dirs:
        for dtype in dtypes:
            try:
                ntk = NTK(chkpath, dtype)
            except Exception:
                # No generated NTK found; move on
                print('Problem with', chkpath, dtype)
                continue
            if ntk.shape[0] > 5000*10:
                continue

            # Get full-path filename of the input file
            filename = ntk.ntkpath + '.zarr'
            outfile = ntk.ntkpath + '_block.npz'
            if os.path.isfile(outfile):
                continue

            print(filename)

            # Get file object
            z = zarr.open(filename, 'r')
            z_bd = block_diag(numpy.array(z))

            # Compute log-determinant (ld) and sign (sign) of the full matrix, as well
            # as the diagonals of U in the LU decomposition.
            try:
                ld, sign, diag, perm, info = detkit.memdet(
                        z_bd, max_mem='32GB', assume='sym', triangle='u',
                        overwrite=False, mixed_precision='float64', parallel_io=None,
                        scratch_dir=scratch_dir, return_info=True, flops=True,
                        verbose=True)
            except ValueError:
                print('Computation failed (likely overflow)')
                continue

            # ------------
            # Write output
            # ------------

            # Save diag and info to file
            numpy.savez(outfile, ld=ld, sign=sign, diag=diag, perm=perm, info=info)


# ===========
# script main
# ===========

if __name__ == "__main__":
    main()
