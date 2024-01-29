import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import sys
import time

#from apex import amp

class trainer:

  def __init__(self, net, train_loader, optimizer, loss_function, logger, 
               tester, test_every,lr_scheduler=None, lrStepPer='batch', 
               preprocImgLbl=None, apex_opt_level=None, update_every=1):
    # net
    # train_loader
    # optimizer
    # loss_function
    # logger
    # tester
    # test_every - a test is run every "test_every" epochs
    # lr_scheduler - 
    # lrStepPer - 'batch' or 'epoch' - make a learning rate update
    #             according to number of batches or epochs
    # preprocImgLbl - a function for batch preprocessing
    #                 I use it to copy the batches to gpu
    self.net=net
    self.dataLoader=train_loader
    self.optimizer=optimizer
    self.crit=loss_function
    self.logger=logger
    self.di=iter(self.dataLoader)
    self.epoch=0
    self.tot_iter=0
    self.prev_iter=self.tot_iter
    self.test_every=test_every
    self.tester=tester
    self.lr_scheduler=lr_scheduler
    self.lrStepPer=lrStepPer # 'batch' or 'epoch'
    self.update_every=update_every
    if preprocImgLbl==None:
      self.preproc=lambda img,lbl: img,lbl
    else:
      self.preproc=preprocImgLbl
#    self.apex_opt_level=apex_opt_level
#    if self.apex_opt_level is not None:
#       self.net,self.optimizer=amp.initialize(self.net,self.optimizer,
#           opt_level=self.apex_opt_level)

  def train(self, numiter):
    self.net.train()
    local_iter=0
    t0=time.time()
    while local_iter<numiter:
      try:
        img, lbl=next(self.di)
        img,lbl=self.preproc(img,lbl)
        if local_iter % self.update_every==0:
          self.optimizer.zero_grad()
        out= self.net.forward(img)
        loss = self.crit(out, lbl)
#        if self.apex_opt_level is not None:
#            with amp.scale_loss(loss,self.optimizer) as scaled_loss:
#                scaled_loss.backward()
#        else:
#            loss.backward()
        loss.backward()
        if local_iter % self.update_every== self.update_every-1:
          self.optimizer.step()
        with torch.no_grad():
          self.logger.add(img,out,lbl,loss.item(),
                          net=self.net,optim=self.optimizer)
        local_iter+=1
        self.tot_iter+=1
        if self.lr_scheduler and self.lrStepPer=='batch':
            self.lr_scheduler.step()
        t1=time.time()
        if t1-t0>3:
          itertime=(t1-t0)/(self.tot_iter-self.prev_iter)
          sys.stdout.write('\rIter: %8d\tEpoch: %6d\tTime/iter: %6f'\
              % (self.tot_iter, self.epoch,itertime))
          t0=t1
          self.prev_iter=self.tot_iter
      except StopIteration:
        lastLoss=self.logger.logEpoch(net=self.net,optim=self.optimizer,
            scheduler=self.lr_scheduler,niter=self.tot_iter)
        self.epoch+=1
        self.di=iter(self.dataLoader)
        if self.test_every and self.epoch%self.test_every==0:
          self.tester.test(self.net)
        if self.lr_scheduler and self.lrStepPer=='epoch':
          self.lr_scheduler.step(lastLoss)
