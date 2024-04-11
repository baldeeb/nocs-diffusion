
## About


## Data
Uses same data as [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://github.com/luost26/diffusion-point-cloud).

## Setup 

Before this is installed as a library, run the following:
```
mamba create -n <env-name> python=<3.8, 3.9, 3.10>
mamba activate
mamba install pytorch=2.1.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath
mamba install pytorch3d -c pytorch3d
pip install -e .
```
NOTE: Torch installation is separate from this package's toml setup is due to [difficulties in pytorch3d setup](https://github.com/facebookresearch/pytorch3d/issues/1419)

