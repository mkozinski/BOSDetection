import torch as th
from torch.nn import Module

def random_select(a,perc,minpts2keep=4):
    # a is a tensor shaped nbatch X nchannels X npoints
    # perc is a scalar in (0,1)
    # randomly select numnz=max(perc*npoints,minpts2keep) points 
    # along the last dimension
    # return a tensor shaped nbatch X nchannels X numnz

    numb=a.shape[0]
    numc=a.shape[1]
    nump=a.shape[-1]

    numnz=min(max(minpts2keep,int(nump*perc)),nump)

    # implement the selection by matrix multiplication

    # 1 prepare the matrix
    mat=th.zeros(nump,numnz)
    r=th.randperm(nump)[0:numnz].unsqueeze(0)
    mat.scatter_(dim=0,index=r,src=th.ones((1,numnz)))

    # 2 multiply
    b=a.reshape(-1,nump)
    c=th.matmul(b,mat.type_as(b).to(b.device))
    return c.reshape(numb,numc,numnz)

class MyMaxPool(Module):
    # max pooling combined with dropping feature vectors
    # (for regularization)
    # difference to ordinary convolutional dropout:
    #  * convolutional dropout drops entire feature planes
    #  * this layer drops individual feature vectors 
    #    it may be followed by ordinary dropout
    # no reweighting is performed
    # -- this works without issues when the next layer is linear
    # -- but might be problematic if a non-linear classifier follows
    #    because the distribution of values output by this layer
    #    will differ in .eval and .training modes

    def __init__(self,percpos=0.5,**kwargs):
        self.percpos=percpos
        super(MyMaxPool,self).__init__(**kwargs)

    def forward(self,x):
        nb,nc=x.shape[0],x.shape[1]
        y=x.reshape(nb,nc,-1)
        if self.training and self.percpos<1.0:
            y=random_select(y,self.percpos)
            
        return th.max(y,dim=-1,keepdim=True)[0]
