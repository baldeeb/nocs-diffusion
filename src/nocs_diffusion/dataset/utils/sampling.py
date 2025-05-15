import torch
from pytorch3d.renderer import look_at_view_transform


# def sample_from_clouds(data, out_count, pad_zeros=True):
#     """ 
#     Point clouds can have unequal number of points. 
#     This functions facilitates making those clouds of equal size.
#     """
#     for i in range(len(data)):
#         pt_count = len(data[i]) 
#         if pt_count < out_count:
#             if pad_zeros:               # Pad with zeros
#                 count_diff = out_count - pt_count
#                 data[i].extend(torch.zeros(count_diff, 3))
#             else:                       # Oversample
#                 pt_idxs = torch.multinomial(torch.ones(out_count), 
#                                             out_count, 
#                                             replacement=True)
#                 data[i] = data[i][pt_idxs]
#         elif pt_count > out_count:      # Undersample
#             data[i] = data[i][torch.randperm(out_count)]
        
#     return torch.stack(data)


###########################################
# TODO: MAKE A VERSION FOR BATCH OF 1

def sample_from_clouds(data, out_count, pad_zeros=True):
    B, N  = data.shape[:2]
    dims  = data.shape[2:]
    if N < out_count:
        out_data = torch.zeros(B, out_count, *dims)
        out_data[:, :N, :] = data
        indices = torch.ones(out_count) * -1
        indices[:, :N] = torch.arange(N)
        if pad_zeros:               # Pad with zeros
            indices[:, N:] = -1
        else:
            pi = ((torch.rand(B, out_count-N) - 1e-6) * N).flatten().long()
            bi = (torch.ones(B, out_count-N) * torch.arange(B)[:, None]).flatten().long()
            indices[:, N:] = pi
            out_data[:, N:] = data[bi, pi]
        data = out_data
    elif N > out_count:
        pi = ((torch.rand(B, out_count) - 1e-6) * N).flatten().long()
        bi = (torch.ones(B, out_count) * torch.arange(B)[:, None]).flatten().long()
        indices = [bi, pi]
        data = data[bi, pi].reshape(B, out_count, *dims)

    return data, indices.long()

def sample_dim_zero(data, out_count, pad_zeros=True):
    '''Data is expected to be N, ...'''
    N  = data.shape[0]
    dims  = data.shape[1:]
    if N < out_count:
        out_data = torch.zeros(out_count, *dims)
        out_data[:N, :] = data
        indices = torch.ones(out_count) * -1
        indices[:N] = torch.arange(N)
        if pad_zeros:               # Pad with zeros
            indices[N:] = torch.nan
        else:
            pi = ((torch.rand(out_count-N) - 1e-6) * N).flatten().long()
            indices[N:] = pi
            out_data[N:] = data[pi]
    else:
        pi = ((torch.rand(out_count) - 1e-6) * N).flatten().long()
        indices = pi
        out_data = data[pi]

    return out_data, indices.long()

def list_to_torch_mat(data, desired_N):
    '''when working with points, it is not always the case that the clouds returned have the
    same point clout. This is meant to deal with the situation by under or over sampling.
    
    input:
        data: List[torch.Tensors] shape B x [Ni x ....]
    output:
        out: torch.Tensor shape B x desired_N x ....
    '''
    B = len(data)
    dims = data[0].shape[1:]

    out = torch.zeros(B, desired_N, *dims)
    out_idxs = torch.zeros(B, desired_N)
    for i in range(B):
        out[i], out_idxs[i] = sample_dim_zero(data[i], desired_N, pad_zeros=False)
    return out, out_idxs.long()


