import numpy as np

from heapq import merge

def FprTprAucFromPredLists(predlist_neg,predlist_pos):
    
    p=[(v,1) for v in sorted(predlist_pos,reverse=True)]
    n=[(v,0) for v in sorted(predlist_neg,reverse=True)]
    preds=merge(p,n,key=lambda v:v[0],reverse=True)
    
    ln=[0,]
    lp=[0,]
    auc=0
    for v in preds:
        ln.append(1-v[1]+ln[-1])
        lp.append(  v[1]+lp[-1])
        auc+=lp[-1]*(1-v[1])
        
    norm =ln[-1]*lp[-1]

    if norm==0:
        return np.zeros(2), np.zeros(2), 0.0

    auc/=float(norm)
    
    tpr=np.array(lp)/float(lp[-1])
    fpr=np.array(ln)/float(ln[-1])
    
    return fpr,tpr,auc

def FprTprAucFromPredLists_Weighted(predlist_neg,predlist_pos):
    # predlists are lists of pairs (value, weight)
    # weights have to be non-negative

    n=sorted(predlist_neg,reverse=True,key=lambda a:a[0])
    # the class of the example will be distinguished by weight sign
    n=[(v,-w) for v,w in n] 
    p=sorted(predlist_pos,reverse=True,key=lambda a:a[0])
    preds=merge(p,n,key=lambda a:a[0],reverse=True)

    ln=[0,]
    lp=[0,]
    auc=0
    for p in preds:
        dneg=max(-p[1],0)
        dpos=max( p[1],0)
        ln.append(dneg+ln[-1]) # negative weight => class 0
        lp.append(dpos+lp[-1]) # positive weight => class 1
        auc+=lp[-1]*dneg
        
    norm =ln[-1]*lp[-1]

    if norm==0:
        return np.zeros(2), np.zeros(2), 0.0

    auc/=float(norm)
    
    tpr=np.array(lp)/float(lp[-1])
    fpr=np.array(ln)/float(ln[-1])
 
    return fpr,tpr,auc
