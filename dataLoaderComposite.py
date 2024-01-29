import torch as th

class DataLoaderComposite():
      

    def __init__(self,dLoaders):
        self.dLoaders=dLoaders
        

    def __len__(self):
        return min([len(dl) for dl in self.dLoaders])


    def __iter__(self):

        it=zip(*self.dLoaders)

        for batchparts in it:

            bpiter=iter(batchparts)
            input_,target=next(bpiter)

            if isinstance(input_,list):
                i=[[inp,] for inp in input_]
            else:
                i=[input_,]

            if isinstance(target,list):
                t=[[trg,] for trg in target]
            else:
                t=[target,]

            for input_,target in bpiter:

                if isinstance(input_,list):
                    for ii,inp in zip(i,input_): ii.append(inp)
                else:
                    i.append(input_)
                    
                if isinstance(target,list):
                    for tt,trg in zip(t,target): tt.append(trg)
                else:
                    t.append(target)

            if isinstance(i[0],list):
                ii=[]
                for inp in i: ii.append(th.cat(inp))
            else:
                ii=th.cat(i)

            if isinstance(t[0],list):
                tt=[]
                for trg in t: tt.append(th.cat(trg))
            else:
                tt=th.cat(t)

            yield(ii,tt)
