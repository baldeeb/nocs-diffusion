import torch 
from torch import nn 
from torch import Tensor
from typing import Optional

class VarianceSchedule(nn.Module):
    def __init__(self, beta_1, beta_T, steps):
        # TODO: summarize or reference theory
        super().__init__()
        self.register_buffer('beta_1', Tensor([beta_1]))
        self.register_buffer('beta_T', Tensor([beta_T]))
        self.register_buffer('steps',  Tensor([steps]))
        betas = torch.linspace(beta_1, beta_T, steps)
        # betas = torch.concatenate([torch.zeros(1), betas])
        alpha_bar = 1 - betas
        for i in range(1, len(alpha_bar)):
            alpha_bar[i] *= alpha_bar[i-1]
        self.register_buffer('noise_ratios', (1 - alpha_bar).sqrt()) 
        self.register_buffer('sample_ratios', alpha_bar.sqrt())

    def sample(self, batch, t=None):
        if t is None:
            t = torch.randperm(self.steps.int().item())[:batch]
        return self.sample_ratios[t], self.noise_ratios[t]


class ForwardDiffuser(nn.Module):
    def __init__(self, var_sched:VarianceSchedule, mean:float=0.0):
        super().__init__()
        self.var_sched = var_sched
        self._mu = mean
    def __call__(self, x:Tensor, noise:Optional[Tensor]=None, t:Optional[int]=None):  
        if noise is None:
            noise = torch.randn_like(x, device=x.device)
        s, n = self.var_sched.sample(noise.shape[0], t)
        s = s[:, None, None, None].to(x.device) 
        n = n[:, None, None, None].to(x.device)
        return x*s + (noise + self._mu)*n
    
class BackwardDiffuser(nn.Module):
    def __init__(self, diffuser:nn.Module):
        super().__init__()
        self._diff = diffuser