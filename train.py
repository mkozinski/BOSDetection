from os import makedirs,path
from shutil import copyfile
from datetime import date
import argparse
import json
from random import randint,  random
from bisect import bisect

import torch
import torch.optim as optim
from torch import nn as nn
import numpy as np

import sys
sys.path.append("../")
from NetworkTraining.loggerBasic import LoggerBasic
from NetworkTraining.loggerF1 import LoggerF1
from NetworkTraining.loggerGeneric import LoggerGeneric
from NetworkTraining.loggerComposite import LoggerComposite
from NetworkTraining.crop import crop
from NetworkTraining.trainer import trainer
from NetworkTraining.dataLoaderComposite import DataLoaderComposite

from net import MyNet

from get_data_loaders import getPrecedenceDataLoader,\
                             getClassifDataLoader

def augmentTrain(trainimg,trainlbl,nslices):
  img=trainimg

  # select nslices random slices 
  inds=torch.randperm(img.shape[0])[:nslices]
  img=img[inds]

  # random crop # note voxel spacing is 0.5,0.5,1
  cropsz=np.array([nslices,512,384],dtype=int) # 256,192,4*48 mm
  #maxstartind1=img.shape[-3]-cropsz[0]
  maxstartind2=img.shape[-2]-cropsz[1]
  maxstartind3=img.shape[-1]-cropsz[2]
  #s1,c1=max(0,maxstartind1),max(0,-maxstartind1)
  s2,c2=max(0,maxstartind2),max(0,-maxstartind2)
  s3,c3=max(0,maxstartind3),max(0,-maxstartind3)
  #startind1=randint(0,s1)-c1//2
  startind1=0
  startind2=randint(0,s2)-c2//2
  startind3=randint(0,s3)-c3//2
  cropinds=[slice(startind1,startind1+cropsz[0]),
            slice(startind2,startind2+cropsz[1]),
            slice(startind3,startind3+cropsz[2]),]
  img,_=crop(img,cropinds,fill=-1)

  # flips in the transverse plane
  if random()>0.5 :
    img=np.flip(img,-1)
  if random()>0.5 :
    img=np.flip(img,-2)

  it=torch.from_numpy(np.copy(img).astype(np.float32))

  lbl    =trainlbl
  lt     =torch.from_numpy(np.copy(lbl) .astype(np.int_))

  return it,lt

def preproc(img,lbl):
  # this function is used in the training loop, and not in the data loader
  # this is needed to avoid transfer of data to GPU in threads of the dataloader
  return img.cuda(), lbl.cuda()

def f1_preproc_classif(output,target,bs_stop):
  # preprocessing for test logger

  o=output[:bs_stop]
  t=target[:bs_stop]

  keep= t>=0
  assert torch.all(keep)

  e=torch.exp(o)
  p=e/(e+1.0)

  return p.cpu(),t.cpu()

def loadfun(fname):
  fullname=path.join(a.root_dir,fname)
  return np.load(fullname,mmap_mode='r')

def precedenceLoss(x,y,min_days_diff):
  x=x.reshape(x.shape[0]//2,2)
  y=y.reshape(y.shape[0]//2,2)
  days_diff=(y[:,1]-y[:,0])

  # this should never happen:
  incorrect = (-min_days_diff < days_diff) & (days_diff < min_days_diff)
  assert torch.all(incorrect==0)

  t = days_diff>=min_days_diff
  ce=nn.functional.cross_entropy(x,t.to(torch.long))
  return ce 

def classifLoss(x,y):
  assert torch.all(y>=0)
  pred=nn.functional.pad(x,(1,0))
  ce=nn.functional.cross_entropy(pred,y)
  return ce

class TotalLoss(nn.Module):
  def __init__(self,min_days_diff,nbatchclassif):
    super(TotalLoss,self).__init__()
    self.nbatchclassif=nbatchclassif
    self.min_days_diff=min_days_diff
  def forward(self,x,y):
    lclassif=classifLoss(x[:self.nbatchclassif],y[:self.nbatchclassif])
    lprec=precedenceLoss(x[self.nbatchclassif:],y[self.nbatchclassif:],
      self.min_days_diff)
    return lclassif+lprec

# batch sizes
bs_classif=10
bs_precedence=10
# number of slices to retain from each scan
nslices=8
# minimum number of days between scans for the precedence task
min_days_diff=182
training_schedule=[50000,   55000,   60000,   65000, ]
learning_rate    =[ 1e-4,(1e-4)/2,(1e-4)/4,(1e-4)/8, ]

if __name__ == '__main__':
  
  p=argparse.ArgumentParser()
  p.add_argument("split_num",type=int)
  p.add_argument("scan_db",type=str)
  p.add_argument("patient_db",type=str)
  p.add_argument("split_map",type=str)
  p.add_argument("root_dir",type=str)
  p.add_argument("log_dir",type=str)
  p.add_argument("--prev_log_dir",type=str)
  a=p.parse_args()
  
  makedirs(a.log_dir)
  copyfile(__file__,path.join(a.log_dir,"setup_script.py"))
  with open(path.join(a.log_dir,"args.json"),"w") as f:
    json.dump(vars(a),f,indent=4)
    f.close()
  
  # the deep network and the loss
  net=MyNet(pretrainedFeatureExtractor=True).cuda()
  optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
  start_iter=0
  start_epoch=0
  
  if a.prev_log_dir is not None:
  
    netname=path.join(a.prev_log_dir,"net_last.pth")
    print("loading network {}".format(netname))
    saved_net=torch.load(netname)
    net.load_state_dict(saved_net['state_dict'])
    start_iter=saved_net['iter']
    start_epoch=saved_net['epoch']
    del saved_net
  
    optfname=path.join(a.prev_log_dir,"optim_last.pth")
    print("loading optimizer {}".format(optfname))
    saved_optim=torch.load(optfname)
    optimizer.load_state_dict(saved_optim['state_dict'])
    del saved_optim
  
  loss=TotalLoss(min_days_diff,bs_classif)
  
  # data loaders
  with open(a.scan_db,"r") as f:
    scan_db=json.load(f)
  
  with open(a.patient_db,"r") as f:
    patient_db=json.load(f)
  
  with open(a.split_map,"r") as f:
    split_map=json.load(f)
  
  classif_dl=getClassifDataLoader(scan_db,patient_db,split_map,a.split_num,
    lambda i,l:augmentTrain(i,l,nslices),bs_classif,loadfun,drop_last=True,
    yieldPatientOncePerEpoch=False)
  precedence_dl=getPrecedenceDataLoader(scan_db,patient_db,split_map,
    a.split_num,lambda i,l:augmentTrain(i,l,nslices),min_days_diff,
    bs_precedence,loadfun,drop_last=True,yieldPatientOncePerEpoch=False)
  train_loader=DataLoaderComposite([classif_dl,precedence_dl])
  
  # logging; save net every 10 epochs
  logger_train_basic = LoggerBasic(a.log_dir,"train_basic",saveNetEvery=10, 
    saveAndKeepEvery=500)
  logger_train_precedence =LoggerGeneric(a.log_dir,"train_precedence",
    lambda i,o,t,l,n,op:precedenceLoss
      (o[bs_classif:].detach(),t[bs_classif:],min_days_diff).cpu().item())
  logger_train_classif    =LoggerF1(a.log_dir,"train_classif",
    lambda i,l:f1_preproc_classif(i,l,bs_classif))
  logger_train=LoggerComposite(
    [logger_train_basic,logger_train_precedence,logger_train_classif,])
  
  print("log directory: {}".format(a.log_dir)

  # training
  trn=trainer(net, train_loader, optimizer, loss, logger_train, None, None,
    lr_scheduler=None,preprocImgLbl=preproc)

  net.train()
  trn.tot_iter=start_iter
  trn.epoch=start_epoch
  logger_train_basic.epoch=start_epoch
  while True:
    target_niter_ind=bisect(training_schedule,trn.tot_iter)
    if target_niter_ind>=len(training_schedule):
      logger_train_basic.save("last.pth", net=trn.net, optim=trn.optimizer,
        niter=trn.tot_iter)
      print(("\ntraining terminates with {} iterations scheduled "
             "and {} iterations executed")\
            .format(training_schedule[-1],trn.tot_iter))
      break
    target_niter =training_schedule[target_niter_ind]
    lr =learning_rate[target_niter_ind]
    iter2run=target_niter-trn.tot_iter
    print(("\niteration {}: setting the learning rate to {} "
           "for the next {} iterations").format(trn.tot_iter,lr,iter2run))
    for pg in trn.optimizer.param_groups :
      pg['lr']=lr
    trn.train(iter2run)
