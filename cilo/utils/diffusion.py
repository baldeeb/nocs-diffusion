import torch 
from torch import nn 

class VarianceSchedule(nn.Module):
    def __init__(self, beta_1, beta_T, steps):
        super().__init__()
        self.register_buffer('beta_1', torch.Tensor([beta_1]))
        self.register_buffer('beta_T', torch.Tensor([beta_T]))
        self.register_buffer('steps', torch.Tensor([steps]))
        betas = torch.linspace(beta_1, beta_T, steps)
        betas = torch.concatenate([torch.zeros(1), betas])
        alphas = 1 - betas
        alpha_bar = alphas.clone()
        for i in range(1, len(alpha_bar)):
            alpha_bar[i] *= alpha_bar[i-1]
        self.register_buffer('noise_ratios', (1 - alpha_bar).sqrt()) 
        self.register_buffer('sample_ratios', alpha_bar.sqrt())

    def sample(self, batch, t=None):
        if t is None:
            t = torch.randperm(self.steps)[:batch]
        return self.sample_ratios[t], self.noise_ratios[t]


class Diffuser(nn.Module):
    def __init__(self, var_sched:VarianceSchedule):
        super().__init__()
        self.var_sched = var_sched
    def __call__(self, x, t=None):  
        noise = torch.rand(x.shape)
        s, n = self.var_sched.sample(noise.shape[0], t)
        return x*s + noise*n