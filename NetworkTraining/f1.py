import torch
import sys
import numpy as np

# routines for computing the precision-recall curves
# and the F1 scores

def reverse(t):
    #idx = [i for i in range(t.size(0)-1, -1, -1)]
    #idx = torch.LongTensor(idx)
    #it  = t.index_select(0, idx)
    it = torch.flip(t,[0])
    return it

def PRFromHistograms(hPos,hNeg):
    positives=hPos.sum()
    negatives=hNeg.sum()
    truepositives =reverse(hPos.clone()).long().cumsum(dim=0)
    falsepositives=reverse(hNeg.clone()).long().cumsum(dim=0)
    predpositives=torch.add(truepositives,falsepositives)
    predpositives[predpositives==0]=1
    precision=torch.Tensor.div(truepositives.float(),predpositives.float())
    recall   =torch.Tensor.div(truepositives.float(),positives.float())
    precision[precision<=0]=sys.float_info.epsilon
    recall   [recall   <=0]=sys.float_info.epsilon
    return precision,recall

def PRFromDiscrete(nTruePos,nPredPos,nGtPos):
    truepositives =nTruePos
    falsepositives=nPredPos-nTruePos
    positives     =nGtPos
    predpositives =nPredPos
    if predpositives==0: predpositives =1 #protect against zero division
    precision=truepositives/float(predpositives)
    recall   =truepositives/float(positives)
    if precision<=0: precision=sys.float_info.epsilon
    if recall   <=0: recall   =sys.float_info.epsilon
    return precision,recall

def PRFromOutputsAndGroundTruths(outps,targs,nbins=10000):
    hPos=torch.zeros(nbins)
    hNeg=torch.zeros(nbins)
    for o,t in zip(outps,targs):
        pos=o[t==1]
        neg=o[t==0]
        hPos+=torch.from_numpy(pos.astype(np.float32)).histc(nbins,0,1)
        hNeg+=torch.from_numpy(neg.astype(np.float32)).histc(nbins,0,1)
    precision,recall=PRFromHistograms(hPos,hNeg)
    return precision,recall

def F1FromPR(p,r):
    suminv=torch.pow(p,-1)+torch.pow(r,-1)
    f1s=torch.pow(suminv,-1).mul(2)
    return f1s


