import numpy as np
import torch
import os.path
import sys

import json

import argparse

sys.path.append("../")
from NetworkTraining.crop import crop

sys.path.append("./")
from net import MyNet

def preproc(testscan):

    img=testscan
#    # if need to save memory, select every n-th slice, for example
#    n=4
#    img=np.ascontiguousarray(img[int(n//2)::n])

    # central crop of the closest size divisible by 16
    divisor=16
    cropsz=(np.floor(np.array(img.shape)/divisor)*divisor).astype(np.int_)
    maxstartind1=img.shape[-3]-cropsz[0]
    maxstartind2=img.shape[-2]-cropsz[1]
    maxstartind3=img.shape[-1]-cropsz[2]
    startind1=maxstartind1//2
    startind2=maxstartind2//2
    startind3=maxstartind3//2
    cropinds=[slice(startind1,startind1+cropsz[0]),
              slice(startind2,startind2+cropsz[1]),
              slice(startind3,startind3+cropsz[2]),]
    img,_=crop(img,cropinds)
  
    it=torch.from_numpy(np.copy(img).astype(np.float32))
  
    return it

def getInput(fname):
    img=np.load(fname)
    img=preproc(img)
    inp=img.reshape(1,img.shape[-3],img.shape[-2],img.shape[-1])
    return inp

def getNet(path,cuda):
    net=MyNet(pretrainedFeatureExtractor=False) 
    state_dict=torch.load(path)["state_dict"]
    net.load_state_dict(state_dict)
    net.eval()
    if cuda:
        net=net.cuda()
    return net
    
def predict(net,inp):
    if next(net.parameters()).is_cuda:
        inp=inp.cuda()
    with torch.no_grad():
        oup=net.forward(inp)
    return oup.cpu()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("netpath",type=str)
    p.add_argument("infile",type=str)
    p.add_argument("--cuda",action="store_true")
    
    a=p.parse_args()
    
    net=getNet(a.netpath,a.cuda)
    
    inp=getInput(a.infile)

    res=predict(net,inp)

    print("{:73} {:>6.2f}".format(a.infile,res.reshape(-1)[0].item()))
    
