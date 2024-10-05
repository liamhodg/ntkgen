# Generating empirical Neural Tangent Kernel matrices
This codebase computes the entire empirical Neural Tangent Kernel matrix for classical image classification datasets, 
defined by
```math
G = (k(x_i, x_j))_{i,j=1}^n \in \mathbb{R}^{mn \times mn},\qquad \mbox{where } k(x, y) = Df(x) Df(y)^\top \in \mathbb{R}^{m \times m},
```
$n$ is the number of distinct elements in the dataset, $m$ is the number of model outputs, and $Df$ is the Jacobian of the network, mapping inputs to matrices of size $\mathbb{R}^{m \times d}$, where $d$ is the number of model parameters.

Support is available for five image classification datasets (using only training data), all of which predict one of $m = 10$ classes:
- CIFAR-10 ($n = 50,000$)
- MNIST ($n = 60,000$)
- KMNIST ($n = 60,000$)
- Fashion-MNIST ($n = 60,000$)
- SVHN ($n = 73,257$)

Note that the complete empirical NTK matrix is an **enormous dense matrix**, and requires significant disk space for storage. Storage requirements for each dataset (independent of model size) with each datatype are as follows:

| Dataset  | float16 | float32 | float64 |
|----------|---------|---------|---------|
| CIFAR-10 | 500 GiB | 1.0 TiB | 2.0 TiB |
| MNIST    | 600 GiB | 1.2 TiB | 2.4 TiB |
| KMNIST   | 600 GiB | 1.2 TiB | 2.4 TiB |
| FMNIST   | 600 GiB | 1.2 TiB | 2.4 TiB |
| SVHN     | 733 GiB | 1.5 TiB | 3.0 TiB |

This code is designed to run even on desktop-class GPUs for smaller ResNets (e.g. ResNet9, ResNet18). Larger models will require significantly more VRAM.

To generate empirical NTK matrices for the float16 and float32 datatypes, simply run:

``python main.py --models {#MODELNAMES} --dataset {#DATASET}``

where MODELNAMES is a space (not comma) separated list of models from:
- resnet9 (4.8M parameters)
- resnet18 (11.1M parameters)
- resnet34 (21.2M parameters)
- resnet50 (23.5M parameters)
- resnet68 (41.4M parameters)
- resnet101 (42.5M parameters)
- resnet152 (58.2M parameters)
- mobilenet (3.2M parameters)
- mobilenetv2 (2.3M parameters)
- vgg11 (9.2M parameters)
- vgg13 (9.4M parameters)
- lenet (62K parameters)
- wrn-28-2 (1.5M parameters)
- wrn-28-5 (9.1M parameters)
- wrn-28-10 (36.5M parameters)
- logistic (30K parameters)
- densenet121 (7.0M parameters)

and DATASET is one of (cifar, mnist, kmnist, fmnist, svhn). Optional arguments for training include

- `--lr` learning rate: default `0.1`
- `--bs` batch size: default `64`
- `--width` ResNet width: default `64`
- `--num_epochs` number of epochs: default `200`
- `--repeats` number of independently trained models: default `1`

Due to the size of the empirical NTK matrix, it is often worthwhile to instead compute the matrix
for a subsample of the dataset. This can be done with the optional argument:

- `--subsample` subsample the dataset for NTK computation: default `None`