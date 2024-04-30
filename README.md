
## About


## Setup 
Get conda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh
```

Before this is installed as a library, run the following:
```
conda create -n <env-name> python=<3.8, 3.9, 3.10>
conda activate <env-name>
conda install pytorch=2.1.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install -e .
```
NOTE: Torch installation is separate from this package's toml setup is due to [difficulties in pytorch3d setup](https://github.com/facebookresearch/pytorch3d/issues/1419)



## Data
Uses same data as [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://github.com/luost26/diffusion-point-cloud). Simply run the commands.
```
cd data
pip install gdown
gdown  https://drive.google.com/uc?id=1WdooIyOMVt_HVd_A0QOM9DTTX41O65jQ
```
