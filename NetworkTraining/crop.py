import numpy as np

def crop(img,inds,fill=0):
    # crop where the range of indexes can contain negative values
    # that is, we can request a crop that goes outside of the image frame
    # img is the numpy array to be cropped
    # inds is a list of slice objects defining the crop
    # negative values denote positions to the left (top) of the image
    # and values larger than image size denote positions to the right (bot)
    # when inds is shorter than img.shape, the initial k missing dimensions
    # are taken to be img.shape[n] where 0<=n<k
    # fill is the value inpainted in the regions outside of the image frame

    outsize=[]
    dstinds=[]
    srcinds=[]

    for k in range(len(inds)):
        i=k+1 # index from the back
        ind=inds[-i]
        shp=img.shape[-i]
        srcbeg=max(0,  ind.start)
        srcend=min(shp,ind.stop)
        dstbeg=max(0, -ind.start)
        dstend=dstbeg+srcend-srcbeg
        dstinds.insert(0,slice(dstbeg,dstend))
        srcinds.insert(0,slice(srcbeg,srcend))
        outsize.insert(0,ind.stop-ind.start)

    # default values for the initial dimensions
    missing_dims=img.ndim-len(inds)
    dstinds=[slice(None)]*missing_dims     +dstinds
    srcinds=[slice(None)]*missing_dims     +srcinds
    outsize=list(img.shape)[0:missing_dims]+outsize

    crop=np.full(tuple(outsize),fill,img.dtype)
    crop[tuple(dstinds)]=img[tuple(srcinds)]

    return crop, dstinds

