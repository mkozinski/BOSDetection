import torch
from torch import nn as nn
from torchvision.models.resnet import resnet18

import sys
sys.path.append("./")
from myMaxPool import MyMaxPool

        
class MyNet(nn.Module):

  def __init__(self, preTrainedFeatureExtractor=False):
    super(MyNet,self).__init__()

    self.feature_extractor=resnet18(pretrained=preTrainedFeatureExtractor)
    # remove the final pooling and fc layers from the resnet
    self.nfeatures=self.feature_extractor.fc.weight.shape[-1]
    self.feature_extractor.fc     =nn.Identity()
    self.feature_extractor.avgpool=nn.Identity()

    # max pooling with randomly dropping voxels
    self.pool=MyMaxPool(0.25)

    self.linear    =nn.Linear(self.nfeatures,1)

  def feature(self,x):
    z=self.feature_extractor(x)
    # separate the channel dimension from the spatial dimension
    # this is needed due to the torchvision representation of ResNet
    # which flattens results to two-dimensions (1:batch,2:everything else) 
    # even when the pooling layer and the fc layer are removed
    z=z.reshape(z.shape[0],self.nfeatures,-1)
    return z

  def forward(self,x):
    nb,d,h,w=x.shape
    # fold the slice (depth) dimension onto the batch dimension
    x2d=x.reshape(nb*d,1,h,w)
    # pretend these are 3-channel, 2D images ...
    x2d3c=x2d.expand(-1,3,-1,-1)
    # ... which can be processed with a resnet feature extractor
    z2d=self.feature(x2d3c)
    # recover the slice(depth) dimension
    nc=z2d.shape[1]
    z3d=z2d.reshape(nb,d,nc,-1)
    # put the slice (depth) dimension in its place
    z3d=z3d.permute(0,2,1,3) 
    # z3d.shape: nbatch,nchannel,depth,height*width

    # pool over the spatial dimensions (the two last dimensions)
    zp=self.pool(z3d).flatten(1)
    
    y=self.linear(zp)
    return y

