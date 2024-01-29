# at times it may be beneficial to process a big block of data (volume, image)
# in little crops, one crop at a time
# since context is important when using a convolutional neural network, 
# we want to ensure that each element of output has been generated 
# with enough context
# thus, when breaking down a large volume into little crops, we should
# generate crops that overlap, and only retain the output elements for each crop
# that have been generated with enough context
# the first crop       XXXXXXXXXX
# the second crop          OOOOOOOOOO
# the outputs combined XXXXXXXOOOOOOO
# 
# this file contains routines for computing the indexes of such crops,
# that overlap with a pre-defined margin, and cover the whole input block
#
# it is used for example by forwardOnBigImages.py
#
#

import math

def noCrops(inSize, cropSize, marginSize, startDim=0):
  # returns the number of crops that fit in a block of inSize
  # inSize 
  # cropSize - can be shorter than inSize, if not all dims are cropped
  #            in this case startDim contains the no of dims 
  #            that are not cropped
  # marginSize - same length as cropSize; stores size of a single margin;
  #              the resulting overlap between crops is 2*marginSize
  # startDim - all dimensions starting from this one are cropped;
  #            for example, if dim 0 indexes batches and dim 1 indexes channels
  #            startDim would typically equal 2;
  #            in this case, cropSize should be 2 elements shorter than inSize 
  #            and only contain crop sizes for dimension 2, 3, 4, etc
  nCrops=1
  for dim in range(startDim, len(inSize)):
    relDim=dim-startDim
    nCropsPerDim=(inSize[dim]-2*marginSize[relDim])/ \
                 (cropSize[relDim]-2*marginSize[relDim])
    if nCropsPerDim<=0:
      nCropsPerDim=1
    nCrops*=math.ceil(nCropsPerDim)
  return nCrops

def noCropsPerDim(inSize,cropSize,marginSize,startDim=0):
  # for call arguments see function noCrops
  # returns:
  # nCropsPerDim - number of crops per dimension, starting from startDim
  # cumNCropsPerDim - number of crops for one index step along a dimension
  #                   starting from startDim-1; i.e. it has one more element
  #                   than nCropsPerDim, and is misaligned by a difference
  #                   in index of 1
  nCropsPerDim=[]
  cumNCropsPerDim=[1]
  for dim in reversed(range(startDim,len(inSize))):
    relDim=dim-startDim
    nCrops=(inSize[dim]-2*marginSize[relDim])/ \
           (cropSize[relDim]-2*marginSize[relDim])
    if nCrops<=0:
      nCrops=1 
    nCrops=math.ceil(nCrops)
    nCropsPerDim.append(nCrops)
    cumNCropsPerDim.append(nCrops*cumNCropsPerDim[len(inSize)-dim-1])
  nCropsPerDim.reverse()
  cumNCropsPerDim.reverse()
  return nCropsPerDim, cumNCropsPerDim

def cropInds(cropInd,cumNCropsPerDim):
  # given a single index of a crop of a given data chunk (image, volume)
  # this function returns crop indeces along all its dimensions
  # these are not image/volume coordinates, 
  # but indeces in the sense "7th crop along dim 1"
  assert cropInd<cumNCropsPerDim[0]
  rem=cropInd
  cropInds=[]
  for dim in range(1,len(cumNCropsPerDim)):
    cropInds.append(rem//cumNCropsPerDim[dim])
    rem=rem%cumNCropsPerDim[dim]
  return cropInds

def coord(cropInd,cropSize,marg,inSize):
# this function maps an index of a volume crop (returned by cropInds)
# to the starting and end coordinate of a crop
# it is meant to be used for a single dimension
# it returns a pair of slices
# the first slice contains the coordinates of the input volume to be cropped
# the second slice contains the coordinates of the target crop
# that contain the central part of the crop without margins.
# if we take only the central parts of all the crops, we can assemble
# back the original volume, without overlaps/repetitions.
# this second pair of coordinates is useful for cropping out
# and stitching together network outputs
# (assuming the network output is the same size as input)
  assert inSize>=cropSize
  startind=cropInd*(cropSize-2*marg) #starting coord of the crop in the big vol
  startValidInd=marg                 #starting coord of valid stuff in crop
  endValidInd=cropSize-marg
  if startind >= inSize-cropSize:
    startValidInd=cropSize+startind-inSize+marg
    startind=inSize-cropSize
    endValidInd=cropSize
  if cropInd==0:
    startValidInd=0
  return slice(int(startind),int(startind+cropSize)), \
         slice(int(startValidInd),int(endValidInd))
         
def coords(cropInds,cropSizes,margs,inSizes,startDim):
# this function maps a table of crop indeces
# to the starting and end coordinates of the crop
# see the comment for "coord" for the description
  cropCoords=[]
  validCoords=[]
  for i in range(startDim):
    cropCoords. append(slice(0,inSizes[i]))
    validCoords.append(slice(0,inSizes[i]))
  for i in range(startDim,len(inSizes)):
    reli=i-startDim
    c,d=coord(cropInds[reli],cropSizes[reli],margs[reli],inSizes[i])
    cropCoords.append(c)
    validCoords.append(d)
  return cropCoords, validCoords

def targetCoords(sourceCoords,validCoords):
  # from coordinates of the crop in the volume,
  # and coordinates of the non-overlapping part of a crop
  # this function computes coordinates of a large output volume,
  # containing network outputs for crops, stitched together,
  # but without the overlapping margins
  cc=sourceCoords
  vc=validCoords
  tc=[]
  for i in range(len(cc)):
    tc.append(slice(cc[i].start+vc[i].start,cc[i].start+vc[i].stop))
  return tc

def cropCoords(cropInd,cropSize,marg,inSize,startDim):
# a single index in, a table of crop coordinates out
# this is the main functionality of this little library
# it allows to extract the coordinates of the n-th crop of a volume
# it returns cropCoords   - that can be used to crop the input volume
#            validCoords  - that can be used to crop the non-overlapping part 
#                           of network output
#            targetCoords - that can be used to stitch the non-overlapping
#                           parts of the crops together
  nCropsPerDim,cumNCropsPerDim=noCropsPerDim(inSize,cropSize,marg,startDim)
  cropIdx=cropInds(cropInd,cumNCropsPerDim)
  cropCoords, validCoords=coords(cropIdx,cropSize,marg,inSize,startDim)
  targCoords = targetCoords(cropCoords,validCoords)
  return cropCoords, validCoords, targCoords

