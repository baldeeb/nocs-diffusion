from pytorch3d.transforms import Transform3d

def to_transform_3d(Rts, scales)->Transform3d:
    """
    Convert a batch of rotation matrices and scales to a batch of Transform3d objects.
    """
    B = Rts.shape[0]
    Ts = Transform3d()
    for i in range(B):
        R = Rts[i, :3, :3]
        t = Rts[i, :3,  3]
        s = scales[i]
        T = Transform3d()
        T.scale(s[0], s[1], s[2])
        T.rotate(R)
        T.translate(t[0], t[1], t[2])
        Ts.stack(T)
    return Ts
