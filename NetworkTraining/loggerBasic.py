import os
import torch

class LoggerBasic:
    def __init__(self, log_dir, name, saveNetEvery=500, saveAndKeep=False, saveAndKeepEvery=None):
        # I changed the interface: saveAndKeepEvery=n implies that 
        # the net will be saved every n epochs under the name net_epoch_<k*n> 
        #
        # I no longer see the need for the "saveNetEvery" argument
        # (as, in practice, I always set it to 1)
        # or the "saveAndKeep" that goes with it
        # but I keep them for backwards compatibility
        self.log_dir=log_dir
        self.log_file=os.path.join(self.log_dir,"log_"+name+".txt")

        text_file = open(self.log_file, "w")
        text_file.close()
        self.loss=0
        self.count=0
        self.saveNetEvery=saveNetEvery
        self.saveAndKeep=saveAndKeep

        self.saveAndKeepEvery=saveAndKeepEvery

        self.epoch=0

    def add(self,img,output,target,l,net=None,optim=None):
        self.loss+=l
        self.count+=1
       
    def save(self, fname, net=None, optim=None, scheduler=None, niter=None):
        if net:
          nfname='net_'+fname
          sdict_cpu={k:v.cpu() for k,v in net.state_dict().items()}
          torch.save({'epoch': self.epoch, 'iter':niter,
                      'state_dict': sdict_cpu},
                     os.path.join(self.log_dir,nfname))
        if optim:
          ofname='optim_'+fname
          torch.save({'epoch': self.epoch, 'iter':niter,
                      'state_dict': optim.state_dict()},
                     os.path.join(self.log_dir,ofname))
        if scheduler:
          ofname='scheduler_'+fname
          torch.save({'epoch': self.epoch, 'iter':niter,
                      'state_dict': scheduler.state_dict()},
                     os.path.join(self.log_dir,ofname))

    def logEpoch(self,net=None,optim=None,scheduler=None,niter=None):
        text_file = open(self.log_file, "a")
        text_file.write(str(self.loss/self.count))
        text_file.write('\n')
        text_file.close()
        lastLoss=self.loss
        self.loss=0
        self.count=0
        self.epoch+=1
        
        if self.saveAndKeepEvery is not None and \
           self.epoch % self.saveAndKeepEvery == 0:
          fname='epoch_'+str(self.epoch)+'.pth'
          self.save(fname,net,optim,scheduler,niter)

        if self.epoch % self.saveNetEvery == 0:
          if self.saveAndKeep:
            fname='epoch_'+str(self.epoch)+'.pth'
          else:
            fname='last.pth'
          self.save(fname,net,optim,scheduler,niter)

        return lastLoss
