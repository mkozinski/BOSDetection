import os
import torch

class LoggerGeneric:
    def __init__(self, log_dir, name, compute_logged_val, 
                 saveMin=False, saveMax=False):
        self.log_dir=log_dir
        self.log_file=os.path.join(self.log_dir,"log_"+name+".txt")
        self.name=name

        self.compute_logged_val=compute_logged_val

        text_file = open(self.log_file, "w")
        text_file.close()
        self.val=0
        self.count=0
        self.epoch=0
        self.min=float("inf")
        self.max=float("-inf")
        self.saveMin=saveMin
        self.saveMax=saveMax

    def add(self,img,output,target,l,net=None,optim=None):
        self.val+=self.compute_logged_val(img,output,target,l,net,optim)
        self.count+=1

    def logEpoch(self,net=None,optim=None,scheduler=None,niter=None):
        text_file = open(self.log_file, "a")
        text_file.write(str(self.val/self.count))
        text_file.write('\n')
        text_file.close()
        lastVal=self.val
        self.val=0
        self.count=0
        self.epoch+=1
        if self.saveMin and lastVal < self.min:
            self.min=lastVal
            if net:
              torch.save({'state_dict':net.state_dict()},
                         os.path.join(self.log_dir,
                         'net_'  +self.name+'_min.pth'))
            if optim:
              torch.save({'state_dict':optim.state_dict()},
                         os.path.join(self.log_dir,
                         'optim_'+self.name+'_min.pth'))
            if scheduler:
              torch.save({'state_dict':optim.state_dict()},
                         os.path.join(self.log_dir,
                         'scheduler_'+self.name+'_min.pth'))
        if self.saveMax and lastVal > self.max:
            self.max=lastVal
            if net:
              torch.save({'state_dict':net.state_dict()},
                         os.path.join(self.log_dir,
                         'net_'  +self.name+'_max.pth'))
            if optim:
              torch.save({'state_dict':optim.state_dict()},
                         os.path.join(self.log_dir,
                         'optim_'+self.name+'_max.pth'))
        
        return lastVal
