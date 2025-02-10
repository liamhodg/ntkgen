#! /usr/bin/env python

import numpy
import detkit
import zarr
import imate
from pprint import pprint
import os
import psutil
import tqdm
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
    dtypes = ['float64']

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
            outfile = ntk.ntkpath + '_slq.npz'
            if os.path.isfile(outfile):
                continue

            print(filename)

            # Get file object
            z = zarr.open(filename, 'r')
            A = numpy.empty(z.shape, dtype=z.dtype)
            # Copy the data from the Zarr array to the NumPy array
            A[:, :] = z[:, :]
            # Fill in lower triangular part
            m = A.shape[0]
            i, j = numpy.tril_indices(m, -1)
            A[i, j] = A[j, i]
            
            x = numpy.linspace(1, ntk.shape[0] // 10, 2)
            x = numpy.floor(x).astype('int32')

            # Compute log-determinant (ld) and sign (sign) of the full matrix, as well
            # as the diagonals of U in the LU decomposition.
            try:
                ld = []
                for idx in tqdm.tqdm(x):
                    B_ = A[:10*idx, :10*idx]
                    B = numpy.copy(B_)
                    res = imate.logdet(B,
                                   gram=False, p=1,
                                   return_info=False, method='slq',gpu=False,
                                   orthogonalize=-1,
                                   num_threads=0)
                    print(res)
                    ld.append(res)
                if numpy.any(numpy.isnan(ld)):
                    print('Fail')
            except ValueError:
                print('Computation failed (likely overflow)')
                continue

            # ------------
            # Write output
            # ------------

            # Save diag and info to file
            numpy.savez(outfile, x=x, ld=numpy.array(ld))


# ===========
# script main
# ===========

if __name__ == "__main__":
    main()
