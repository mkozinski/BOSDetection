class LoggerComposit:
    def __init__(self, loggers):
        self.loggers=loggers

    def add(self,img,output,target,l,net=None,optim=None):
        for lgr in self.loggers:
            lgr.add(img,output,target,l,net,optim)

    def logEpoch(self,net=None,optim=None,scheduler=None,niter=None):
        lastLoss=self.loggers[0].logEpoch(net,optim,scheduler,niter)
        for k in range(1,len(self.loggers)):
            self.loggers[k].logEpoch(net,optim,scheduler,niter)
        return lastLoss
