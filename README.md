
## About


## Data
Uses same data as [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://github.com/luost26/diffusion-point-cloud).

## Setup 

Before this is installed as a library, run the following:
```
mamba create -n <env-name> python=<3.8, 3.9, 3.10>
mamba activate
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -e .
```
NOTE: Torch installation is separate from this package's toml setup is due to [difficulties in pytorch3d setup](https://github.com/facebookresearch/pytorch3d/issues/1419)

