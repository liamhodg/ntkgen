import zarr
from tqdm import tqdm, trange
import numpy as np
import shutil

def inv_block(mat, block_size, num_blocks, save=None):
    """Compute the first K x K block of the inverse of a given
    matrix using recursive block matrix inversion, where K
    is input as block_size and the total size of mat is
    block_size x num_blocks. Support for automatic save of the
    result into a npy file with path given by the save
    parameter."""
    total_size = block_size * num_blocks
    assert mat.shape[0] == total_size
    
    # Copy all data into temporary zarr file
    tmp = zarr.create_array(
        store="tmp.zarr",
        shape=(total_size, total_size),
        chunks=(num_blocks, num_blocks),
        dtype="float64"
    )
    print('- Copying into temporary array...')
    pbar_transfer = tqdm(total=num_blocks**2)
    for idx in range(num_blocks):
        for idy in range(num_blocks):
            R = mat[idx*block_size:(idx+1)*block_size,
                    idy*block_size:(idy+1)*block_size]
            tmp[idx*block_size:(idx+1)*block_size,
                idy*block_size:(idy+1)*block_size] = R
            pbar_transfer.update(1)
    del pbar_transfer

    # Perform recursive Schur decomposition
    b1 = num_blocks - 1
    total_steps = int(b1*(b1+1)*(2*b1+1) / 6)
    pbar = tqdm(total=total_steps)
    for block in range(num_blocks,1,-1):
        # Reduce each time by K
        D = tmp[(block-1)*block_size:block*block_size, 
                (block-1)*block_size:block*block_size]
        D_inv = np.linalg.inv(D)
        for idx in range(block-1):
            B = tmp[idx*block_size:(idx+1)*block_size,
                    (block-1)*block_size:block*block_size] @ D_inv
            for idy in range(block-1):
                C = tmp[(block-1)*block_size:block*block_size,
                        idy*block_size:(idy+1)*block_size]
                tmp[idx*block_size:(idx+1)*block_size,
                    idy*block_size:(idy+1)*block_size] -= B @ C
                pbar.update(1)

    res = np.linalg.inv(tmp[:block_size,:block_size])
    # Remove temporary files
    del tmp
    shutil.rmtree('tmp.zarr')
    if save is not None:
        np.save(save, res)
    return res