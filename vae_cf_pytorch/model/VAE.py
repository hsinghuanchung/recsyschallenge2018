import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,p_dims,q_dims=None):
        super(VAE,self).__init__()
        if(q_dims == None):
            q_dims = p_dims[::-1]
            
        self.enlayer1 = nn.Linear(q_dims[0],q_dims[1],bias=True)
        self.enlayer_out1 = nn.Linear(q_dims[1],q_dims[2],bias=True)
        self.enlayer_out2 = nn.Linear(q_dims[1],q_dims[2],bias=True)

        self.test1 = nn.Linear(p_dims[0],p_dims[1],bias=True)
        self.test2 = nn.Linear(p_dims[1],p_dims[2],bias=True)
    def encode(self,x):
        out = F.tanh(self.enlayer1(x))
        self.mean = self.enlayer_out1(out)
        self.var = self.enlayer_out2(out)
        return self.mean,self.var

    def reparametric(self):
        if(self.training == True):
            std = torch.exp(0.5*self.var)
            eps = torch.randn_like(std).cuda()
            sam = eps.mul(std).add_(self.mean)
        else:
            sam = self.mean
            
        return sam

    def decode(self,x):
        out = F.tanh(self.test1(x))
        out = self.test2(out)
        
        return out

    def forward(self,x,training=True):
        self.training = training
        
        self.encode(x)
        sam = self.reparametric()
        out = self.decode(sam)

        return out,self.mean,self.var
    
    def loss_function(self,recon_x, x, mu, logvar,anneal=1):
        recon_x = F.log_softmax(recon_x,dim=1)
        
        CE = -torch.sum(recon_x*x,dim=1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
        
        return torch.mean(CE + anneal * KLD)