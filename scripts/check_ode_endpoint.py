#!/usr/bin/env python3
import torch
from src.sde.freqsde import SDE
from src.sde import get_ode_sampler

class DummySDE(SDE):
    feature_dim=1
    real_n_fft=None
    def __init__(self): super().__init__(1.0,10)
    def sde(self,x,y,t):
        B,D=x.shape
        return torch.ones_like(x), torch.zeros(B,D,D,dtype=x.dtype,device=x.device)
    def marginal_prob(self,x,y,t): raise NotImplementedError
    def prior_sampling(self,shape,*args): return torch.zeros_like(args[0])
    def prior_logp(self,z): raise NotImplementedError
    @staticmethod
    def add_argparse_args(p): return p
    def copy(self): return DummySDE()

def score(x,y,t): return torch.zeros_like(x)

def main():
    y=torch.zeros(1,2,dtype=torch.complex64)
    eps=.005
    sampler=get_ode_sampler(DummySDE(),score,y,denoise=False,eps=eps,device='cpu',rtol=1e-8,atol=1e-8)
    x,nfe=sampler()
    err=float((x.real-(eps-1)).abs().max())
    print('nfe',nfe,'err',err)
    assert err<1e-5
    print('PASS ode')
if __name__=='__main__': main()
