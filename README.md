
## About
Training a diffusion based Normalized-Object-Coordinate-System discriminator.

**Backstory**: A while back I authored a workshop paper suggesting that one can train a NOCS discriminator that is not conditioned on the input domain. Doing so effectively could allow one to train an RGB conditioned NOCS predictor without the need for labeling over all input domains; One can train a NOCS discriminator using labeled simulated data then use that to train a predictor for other unlabeled input domains (real images). Obviously there are caviates, primarily that the discriminator ought to be conditioned on data that can be made very similar in all target (labeled and unlabeled domains). That discriminator conditioning can be a class label or depth related data for example.

**Motivation**: The number one question I got about my prior work was why it did not leverage diffusion. At the time, adversarial training seemed key for that work. This work is an exploration of the potential for using diffusion towards that end.

**Progress Posts:**: pending 

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
