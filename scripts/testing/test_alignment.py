import torch
from pytorch3d.transforms import Transform3d
from nocs_diffusion.utils.transforms import to_transform_3d
from nocs_diffusion.utils.alignment import simple_align

def get_random_transform_3d():
    Ts = Transform3d()
    rand = lambda : torch.rand(1).item()
    for _ in range(B):
        t = Transform3d()
        t.scale(rand() * 3.0)
        t.translate(rand()*5.0, rand()*5.0, rand()*5.0)
        t.rotate_axis_angle(axis='X', angle=rand()*360)
        t.rotate_axis_angle(axis='Y', angle=rand()*360)
        t.rotate_axis_angle(axis='Z', angle=rand()*360)
        Ts.stack(t)
    return Ts
    

if __name__=='__main__':
    B = 4
    N = 100

    # create a set of 3d points
    pts = torch.rand(B, N, 3)

    # create transforms 
    Tfs = get_random_transform_3d()
    tf_pts = Tfs.transform_points(pts)

    Rt, corr_loss, dist_PQ, scale = simple_align(tf_pts, pts)
    Ts_pred = to_transform_3d(Rt, scale)

    print(f'num transforms in Ts: {len(Tfs)}')
    print(Tfs.get_matrix() @ Ts_pred.get_matrix())