from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import torch as th

from datetime import date

from random import shuffle,randrange
from copy import deepcopy
from itertools import chain

import sys
from NetworkTraining.simpleDataset import SimpleDataset

class SamplerPairsPosNeg(Sampler):

    def __init__(self, indList, 
                 ind2label,
                 ind2scanner, 
                 ind2patient,
                 data_source,
                 yieldPatientOncePerEpoch=False,
                 *args, **kwargs):
    
        super(SamplerPairsPosNeg,self).__init__(data_source, *args,**kwargs)

        self.ind2patient=ind2patient
        self.yieldPatientOncePerEpoch=yieldPatientOncePerEpoch

        self.scanner2indList={}
        for i in indList:
            lbl    =ind2label  [i]
            scanner=ind2scanner[i]
            if scanner not in self.scanner2indList:
                self.scanner2indList[scanner]=([],[])
            self.scanner2indList[scanner][lbl].append(i)

 
    def __len__(self):
        totlen=0
        for il in self.scanner2indList.values():
            totlen+=min(len(il[0]),len(il[1]))
        totlen=2*totlen
        return totlen


    def generatePairs(self):

        pairs=[]
        for il in self.scanner2indList.values():
            shuffle(il[0])
            shuffle(il[1])
            pairlist=list(zip(il[0],il[1]))
            pairs=pairs+pairlist

        shuffle(pairs)
        return pairs

    def generatePairsUniquePatients(self):

        included_patients_0=set()
        included_patients_1=set()
        pairs=[]
        for il in self.scanner2indList.values():
            il0=[]
            shuffle(il[0])
            for ind in il[0]:
                if self.ind2patient[ind] not in included_patients_0:
                    il0.append(ind)
                    included_patients_0.add(self.ind2patient[ind])
            il1=[]
            shuffle(il[1])
            for ind in il[1]:
                if self.ind2patient[ind] not in included_patients_1:
                    il1.append(ind)
                    included_patients_1.add(self.ind2patient[ind])
    
            pairlist=list(zip(il0,il1))
            pairs=pairs+pairlist

        shuffle(pairs)
        return pairs

    def __iter__(self):

        if self.yieldPatientOncePerEpoch:
            return chain.from_iterable(self.generatePairsUniquePatients())
        else:
            return chain.from_iterable(self.generatePairs())

class SamplerPairsSamePatSameScanner(Sampler):

    def __init__(self, indList, ind2patient,
                 ind2scanner, ind2scandate, minimumdiff,
                 data_source, yieldPatientOncePerEpoch=False, 
                 *args, **kwargs):

        # indsList     - list of scan indices
        # ind2patient  - dict int:patientName
        # ind2scanner  - dict int:scannerString
        # ind2scandate - dict int:int (days since a fixed date)
        # minimumdiff  - int - minimum number of days between two scans
        #                to be paired
        # data_source  - torch.utils.data.Dataset
        # yieldPatientOncePerEpoch - bool:yield only one scan of each patient 
        #                            per epoch

        super(SamplerPairsSamePatSameScanner,self)\
             .__init__(data_source, *args,**kwargs)

        self.scandate=deepcopy(ind2scandate)
        self.mindiff =minimumdiff
        self.scaninds=indList
        self.ind2patient=ind2patient
        self.yieldPatientOncePerEpoch=yieldPatientOncePerEpoch

        # identify groups of scans that can be paired:
        # ones of the same patient and taken with the same scanner
        scanner2pat2indlist={} 
        for i in self.scaninds:
            scanner=ind2scanner[i]
            patient=ind2patient[i]
            if scanner not in scanner2pat2indlist:
                scanner2pat2indlist[scanner]={}
            if patient not in scanner2pat2indlist[scanner]:
                scanner2pat2indlist[scanner][patient]=[]
            scanner2pat2indlist[scanner][patient].append(i)

        # prepare lists of possible scan pairs
        self.possiblePairs=[] # each element is a list of pairs 
        for pat2indlist in scanner2pat2indlist.values():
            for indlist in pat2indlist.values():
                self.possiblePairs.append(self.getPossiblePairs(indlist))

    def getPossiblePairs(self,scans):
        d=th.tensor([self.scandate[s] for s in scans])
        dif=d.unsqueeze(0)-d.unsqueeze(1) # matrix of days between two scans
        pair_candidates=dif>=self.mindiff # scan1 mindiff days after scan0
        pair_cand_inds=th.nonzero(pair_candidates)
        pair_cand_list=list(pair_cand_inds)
        return [(scans[s[0]],scans[s[1]]) for s in pair_cand_list]
        
    def __len__(self):
        # this is not the true number of elements
        # the true number of elements differs epoch to epoch
        # (this method is not used)
        totlen=0
        for l in self.pariableInds.values():
          totlen+=len(l)//2 *2
        return totlen

    def randomlySelectPairs(self,pairList):
        shuffle(pairList)

        used_scans=set()
        scan_pairs=[]
        for pair in pairList:
            # do not repeat a scan in the epoch
            if pair[0] not in used_scans and pair[1] not in used_scans:
                used_scans.add(pair[0])
                used_scans.add(pair[1])
                scan_pairs.append((pair[0],pair[1]))
        return scan_pairs

    def generatePairs(self):

        pairs=[]
        for pair_list in self.possiblePairs:
            pairs=pairs+self.randomlySelectPairs(pair_list)

        shuffle(pairs)

        return pairs

    def generatePairsUniquePatients(self):
        # each patient is seen once per epoch

        pairs=[]
        includedPatients=set()
        for pair_list in self.possiblePairs:
            if len(pair_list)>0:
                newPair=pair_list[randrange(0,len(pair_list))]
                pat=self.ind2patient[newPair[0]]
                if pat not in includedPatients:
                    pairs.append(newPair)
                    includedPatients.add(pat)

        shuffle(pairs)

        return pairs
       
    def __iter__(self):

        if self.yieldPatientOncePerEpoch:
            pairs=self.generatePairsUniquePatients()
        else:
            pairs=self.generatePairs()

        assert len(pairs)>0

        return chain.from_iterable(pairs)

def getClassifDataLoader(scan_db, patient_db, split_map, split, augmentFun,
    bs,loadFun,drop_last=True,yieldPatientOncePerEpoch=None):
    # list of scans of BOS patients

    def after_bos_onset(scan):
        pat=scan_db[scan]["patient"]
        if "0.8" not in patient_db[pat]["FEV1_level_dates"]:
            return False
        onset_date=patient_db[pat]["FEV1_level_dates"]["0.8"]
        scan_date =scan_db[scan]["date"]
        return (date(**scan_date)-date(**onset_date)).days>=0

    # list of scans of patients without BOS 
    # and scans of patients with BOS taken after BOS onset
    scanList=[scan for scan in scan_db.keys() \
              if split_map [scan_db[scan]["patient"]]!=split and \
              (patient_db[scan_db[scan]["patient"]]["label"]==0 or \
               after_bos_onset(scan))]
    scanLbl=[patient_db[scan_db[scan]["patient"]]["label"] \
        for scan in scanList]
    train_dataset=SimpleDataset(scanList,scanLbl,augmentFun,get_img=loadFun)

    indList=list(range(len(scanList)))
    ind2label   ={i:l                        for i,l    in enumerate(scanLbl)}
    ind2scanner ={i:scan_db[scan]["scanner"] for i,scan in enumerate(scanList)} 
    ind2patient ={i:scan_db[scan]["patient"] for i,scan in enumerate(scanList)}
    sampler_train=SamplerPairsPosNeg(indList,ind2label, ind2scanner,
        ind2patient,train_dataset,
        yieldPatientOncePerEpoch=yieldPatientOncePerEpoch)
    train_loader = DataLoader(train_dataset,batch_size=bs,
        sampler=sampler_train,num_workers=bs,drop_last=drop_last)

    return train_loader

def getPrecedenceDataLoader(scan_db, patient_db, split_map, split, augmentFun, 
    days_diff, bs, loadFun, drop_last=True, yieldPatientOncePerEpoch=None):
    # this data loader produces pairs of scans of the same patient
    # and taken with the same scanner, at least days_diff days apart
    # it includes all patients except the ones in test split 
    # specified by the "split" argument

    # list of scans of BOS patients
    scanList=[scan for scan in scan_db.keys() \
              if  patient_db[scan_db[scan]["patient"]]["label"]==1 \
              and split_map [scan_db[scan]["patient"]]!=split ]
    d0=date(year=1990,month=1,day=30)
    scanLbl=[(date(**scan_db[scan]["date"])-d0).days for scan in scanList]
    train_dataset=SimpleDataset(scanList,scanLbl,augmentFun,get_img=loadFun)

    indList=list(range(len(scanList)))
    ind2scanner ={i:scan_db[scan]["scanner"] for i,scan in enumerate(scanList)}
    ind2patient ={i:scan_db[scan]["patient"] for i,scan in enumerate(scanList)}
    ind2scantime={i:l                        for i,l    in enumerate(scanLbl)}
    sampler_train=SamplerPairsSamePatSameScanner(indList,
        ind2patient,ind2scanner,ind2scantime, days_diff, train_dataset,
        yieldPatientOncePerEpoch=yieldPatientOncePerEpoch)
    train_loader = DataLoader(train_dataset,batch_size=bs,
        sampler=sampler_train,num_workers=bs,drop_last=drop_last)

    return train_loader

