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
import detkit
import zarr
from pprint import pprint
import os
import psutil
from ntk import NTK


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
    dtypes = ['float16','float32','float64']

    # Set available memory for MEMDET computation to 80% of available RAM
    mem_avail = int(psutil.virtual_memory().available * 0.8)

    # mixed precision: str {'float32', 'float64'} or None
    # the precision at while the computations are performed.
    # It is recommended to set a precision higher than the dtype of the input
    # If None, the precision of the input data is used.
    mixed_precision = 'float64'

    # scratch_dir: str or None
    # The directory where memdet will create temporary scratch file. If None,
    # the default OS's tmp directorty will be used. In Linux, this is '/tmp'.
    # This directory should have an available space of at least the size of the
    # input matrix.
    # Note: Often '/tmp' directory is not assigned a large space. Make sure
    # '/tmp' has enough space, otherwise, specify another directory other than
    # leaving this argument as None.
    scratch_dir = 'tmp'

    # -----------
    # Computation
    # -----------
    for chkpath in dirs:
        for dtype in dtypes:
            try:
                ntk = NTK(chkpath, dtype)
            except Exception:
                # No generated NTK found; move on
                continue

        # Get full-path filename of the input file
        filename = ntk.ntkpath + '.zarr'

        # Get file object
        z = zarr.open(filename, 'r')

        # Compute log-determinant (ld) and sign (sign) of the full matrix, as well
        # as the diagonals of U in the LU decomposition.
        ld, sign, diag, info = detkit.memdet(
                z, max_mem=mem_avail, assume='sym', triangle='u',
                overwrite=False, mixed_precision=mixed_precision, parallel_io=None,
                scratch_dir=scratch_dir, return_info=True, flops=True,
                verbose=True)

        # -------------
        # Print results
        # -------------

        # Check the log-determinant (ld) should be the same as the sum of log of
        # all diagonals (diag)
        print(ld)
        print(numpy.sum(numpy.log(numpy.abs(diag))))
        print(sign)

        # Print computation information
        pprint(info)

        # ------------
        # Write output
        # ------------

        # Save diag and info to file
        numpy.savez(ntk.ntkpath + '_logdet.npz', diag=diag, info=info)


# ===========
# script main
# ===========

if __name__ == "__main__":
    main()
