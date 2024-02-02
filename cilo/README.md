## TODO:
 -[] Add controlls in renderer to: 
    -[] turn on off noise
    -[] add noise to vertices
    -[] pass in pose generator rather than having it live in renderer. I might want to reuse a series of poses that are similarn 
 -[] Build data pipeline: pass points to renderer, add noises and apply poses, then save the series of diffused images 

## installation 
- python 3.8
- pytorch 2.0
- (pytorch3d)[https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md]: conda install pytorch3d -c pytorch3d
- conda install h5py