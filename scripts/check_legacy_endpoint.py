#!/usr/bin/env python3
import torch
from src.sde.freqsde import SDE
from src.sde import get_pc_sampler

class DummySDE(SDE):
    feature_dim = 1
    real_n_fft = None
    def __init__(self, T=1.0, N=10):
        super().__init__(T, N)
    def sde(self, x, y, t):
        B,D=x.shape
        return torch.ones_like(x), torch.zeros(B,D,D,dtype=x.dtype,device=x.device)
    def marginal_prob(self,x,y,t):
        raise NotImplementedError
    def prior_sampling(self,shape,*args):
        y=args[0]
        return torch.zeros_like(y)
    def prior_logp(self,z):
        raise NotImplementedError
    @staticmethod
    def add_argparse_args(p): return p
    def copy(self): return DummySDE(self.T,self.N)

def score(x,y,t): return torch.zeros_like(x)

def main():
    eps=.005
    y=torch.zeros(2,3,dtype=torch.complex64)
    for N in [5,10,20,50,100,500]:
        sde=DummySDE(N=N)
        sampler=get_pc_sampler('reverse_diffusion','none',sde,score,y,denoise=True,eps=eps,probability_flow=True)
        x,nfe=sampler()
        err=float((x.real-(eps-1)).abs().max())
        print(N,nfe,err)
        assert err < 2e-5
    print('PASS legacy')
if __name__=='__main__': main()
