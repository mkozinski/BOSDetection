import torch
from torch.utils.data import Dataset
import numpy as np
import NetworkTraining_py.cropRoutines as cropRoutines
from NetworkTraining_py.crop import crop
import bisect

class TestDataset(Dataset):
# this dataset enables testing a segmentation network on a large volume/image
# by cutting the volume or image into overlapping crops/tiles
# but only retaining non-overlapping ground truth chunks 
# so that each pixel/voxel is evaluated once

  def __init__(self, img, lbl, cropSz, margSz, augment,
               ignoreInd=255, output_crop_inds=False):
    self.cropSz=cropSz
    self.margSz=margSz
    self.img=img
    self.lbl=lbl
    self.no_crops=[0] # no_crops[i] = sum no crops in img[k] for all k<i
    self.ignoreInd=ignoreInd 
    self.output_crop_inds=output_crop_inds
    self.augment=augment
    for l in self.lbl:
      tot_no_crops=cropRoutines.noCrops(l.shape,self.cropSz,self.margSz,0)+\
                   self.no_crops[-1]
      self.no_crops.append(tot_no_crops)
  
  def __len__(self):
    return self.no_crops[-1]

  def getCrop(self,idx):
    # idx is a crop index, not an image index
    ind=bisect.bisect_right(self.no_crops,idx)-1  # image index
    lbl=self.lbl[ind]
    img=self.img[ind]
    cropInd=idx-self.no_crops[ind] # index of crop of img[ind]
    cc,vc,_=cropRoutines.cropCoords(cropInd,self.cropSz,self.margSz,lbl.shape,0)
    cc=tuple(cc)
    cimg,_=crop(img,cc)
    clbl=lbl[cc].copy()
    # use vc to inpaint margins to ignore in lbl
    idx=[]
    for i in range(len(vc)):
      # left margin
      idx.append(slice(0,vc[i].start))
      clbl[tuple(idx)]=self.ignoreInd
      del idx[-1]
      # right margin
      idx.append(slice(vc[i].stop,clbl.shape[i]))
      clbl[tuple(idx)]=self.ignoreInd
      del idx[-1]
      # prepare index for next dimensions
      idx.append(slice(0,clbl.shape[i]))
     
    if self.output_crop_inds:
      cinds=np.array([[ind.start,ind.stop] for ind in vc])
      clbl=[clbl,cinds]

    return cimg, clbl
    

  def __getitem__(self, idx):
    img,lbl=self.getCrop(idx)
    img,lbl=self.augment(img,lbl)
    return (img, lbl)

