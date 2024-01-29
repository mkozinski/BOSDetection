import numpy as np
import NetworkTraining_py.cropRoutines as cropRoutines
import torch

def processBigInput(inChunk,cropSize,marginSize,startDim,func,outDims=None):
  # process a big input image or volume, in crops (crop-by-crop)
  # inChunk - the input volume
  # cropSize - the size of the crop to be processed at a time, a tuple
  # marginSize-a tuple, margin size along eqch dimension
  # startDim - int, the index of the first dimension to crop; 
  #            the input is not cropped along dimensions lower than startDim
  # func     - the function to be executed on the crops
  # outDims-a tuple; the size of the output along k first dimensions along which
  # there is no cropping; (network output can have different number of channels
  # than the input)
  nc=cropRoutines.noCrops(inChunk.shape,cropSize,marginSize,startDim)
  size=inChunk.shape
  if outDims:
    size=[]
    for k in range(len(outDims)):
      size.append(outDims[k])
    for k in range(startDim,inChunk.ndim):
      size.append(inChunk.shape[k])
  outChunk=np.zeros(tuple(size))
  for i in range(nc):
    cc,vc,tc=cropRoutines.cropCoords(i,cropSize,marginSize,inChunk.shape,startDim)
    crop=inChunk[tuple(cc)]
    o=func(crop)
    if outDims:
      ttc=[]
      tvc=[]
      for k in range(len(outDims)):
        ttc.append(slice(0,outDims[k]))
        tvc.append(slice(0,outDims[k]))
      for k in range(startDim,inChunk.ndim):
        ttc.append(tc[k])
        tvc.append(vc[k])
      tc=ttc
      vc=tvc
    outChunk[tuple(tc)]=o[tuple(vc)]
  return outChunk
