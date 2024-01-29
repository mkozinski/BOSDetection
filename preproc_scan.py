import argparse
from sys import exit
from os.path import join,isfile
import json
from operator import itemgetter

import numpy as np
from scipy.ndimage import zoom

from pydicom import dcmread
from pydicom.errors import InvalidDicomError

from datetime import date

def preselect(dcm):
    if dcm.Modality!="CT":
        return False
    if dcm.noImgs < 50:
        return False
    if dcm.PixelSpacing[0]>=0.75:
        return False
## check the description for keywords
#    if "THORA" not in dcm.StudyDescription:
#        return False
#    if "ANGIO" in dcm.StudyDescription or "Angio" in dcm.StudyDescription \
#    or "EMBO"  in dcm.StudyDescription or "Embo"  in dcm.StudyDescription \
#    or "PERF"  in dcm.StudyDescription or "Perf"  in dcm.StudyDescription \
#    or "COER"  in dcm.StudyDescription or "Coer"  in dcm.StudyDescription \
#    or "CARD"  in dcm.StudyDescription or "Card"  in dcm.StudyDescription:
#        return False

    return True

def orderInstances(instance_list):
    # takes as argument a list of dicom objects
    # returns the same list, ordered along the scanning direction
    if len(instance_list)<2:
        return instance_list
    pos0=np.array(instance_list[0].ImagePositionPatient)
    pos1=np.array(instance_list[1].ImagePositionPatient)
    slice_vec=(pos1-pos0)
    # the position of consecutive slices should be changing at least along
    # one dimension
    keyind=int(np.argmax(np.fabs(slice_vec)))
    p0=pos0[keyind]
    pos=[inst.ImagePositionPatient[keyind]-p0 for inst in instance_list]
    s_inst=[i for _,i in sorted(zip(pos,instance_list),key=itemgetter(0))]
    return s_inst

def constructVolume(instance_list,ordered=True):
    # returns a volume with coordinate order: slice-row-col
    # (under the assumption that dicom.PixelArray gives row-col)
    if ordered:
        s_inst=instance_list
    else:
        s_inst=orderInstances(instance_list)
    scan=np.stack([i.pixel_array for i in s_inst ],axis=0)
    return scan

def rescaleSlices(vol,currpixsize,targetpixsize):
    # rescale each slice of vol independently
    vols=[]
    zoomfac=currpixsize/targetpixsize
    for i in range(vol.shape[-1]):
        vols.append(zoom(vol[:,:,i],zoomfac,order=1,mode='nearest'))
    return np.stack(vols,axis=-1)

def transformVolume(vol,rowcosxyz,colcosxyz):
    # normally, patients are roughly aligned with a scanner
    # so this does not need to work for arbitrary rotations
    # only for axis permutations and flips
    #
    # assume vol has dimensions: slice - rows - columns; in short: SRC;
    # that is, the dicom images were stacked along axis 0
    #
    arowcosxyz=np.abs(rowcosxyz)
    acolcosxyz=np.abs(colcosxyz)
    col_dim=np.argmax(arowcosxyz) # along this dim the cols change
                                  # note this is NOT the dimension along a col!
    row_dim=np.argmax(acolcosxyz) # along this dim the rows change
                                  # note this is NOT the dimension along a row!
    assert (row_dim!=col_dim)
    slc_dim=3-row_dim-col_dim # along this dim the slices are stacked

    # flip the images if direction is reverted
    # this is only needed along rows and cols,
    # because slices are ordered elsewhere
    vof=vol
    if rowcosxyz[col_dim]<0:
        vof=np.flip(vof,axis=2)
    if colcosxyz[row_dim]<0:
        vof=np.flip(vof,axis=1)

    # permute the dimensions to XYZ
    src2xyz=np.array([slc_dim,row_dim,col_dim],dtype=np.uint)
    xyz2src=np.zeros(3,dtype=np.uint)
    xyz2src[src2xyz]=[0,1,2]
    vop=np.transpose(vof,xyz2src)

    return vop, src2xyz,xyz2src

def getTransformedVolume(instance_list):
    v=constructVolume(instance_list,ordered=True)

    # get the orientation of the slice, pixel size and distance between slices
    # then transform the geometry
    iop=np.array(instance_list[0].ImageOrientationPatient) # row and col cosines with axes
    nv,src2xyz,xyz2src=transformVolume(v,iop[0:3],iop[3:6])

    # remap the intensity
    slope  =instance_list[0].RescaleSlope
    intrcpt=instance_list[0].RescaleIntercept
    nv=nv*slope+intrcpt

    return nv,src2xyz,xyz2src

def getVoxelSpacingXYZ(oins,xyz2src):
    ps=oins[0].PixelSpacing # pixel size
    interslc=np.max(np.array(oins[1].ImagePositionPatient)
                   -np.array(oins[0].ImagePositionPatient)) # inter-slice dist
    spacingsrc=np.array([interslc,ps[0],ps[1]])
    return spacingsrc[xyz2src]

def report_repeated(dcm1,dcm2):
    
    print("repetition")
    
    dcm=dcm1
    suid=dcm.SeriesInstanceUID if hasattr(dcm,"SeriesInstanceUID") else "not present"
    sn  =dcm.SeriesNumber      if hasattr(dcm,"SeriesNumber")      else "not present"
    an  =dcm.AcquisitionNumber if hasattr(dcm,"AcquisitionNumber") else "not present"
    iuid=dcm.SOPInstanceUID    if hasattr(dcm,"SOPInstanceUID")    else "not present"
    inr =dcm.InstanceNumber    if hasattr(dcm,"InstanceNumber")    else "not present"
    print("\t\tseries UID: {}, series: {}, acquisition: {}, instance UID: {}, instance: {}".format(suid,sn,an,iuid,inr))
    
    dcm=dcm2
    suid=dcm.SeriesInstanceUID if hasattr(dcm,"SeriesInstanceUID") else "not present"
    sn  =dcm.SeriesNumber      if hasattr(dcm,"SeriesNumber")      else "not present"
    an  =dcm.AcquisitionNumber if hasattr(dcm,"AcquisitionNumber") else "not present"
    iuid=dcm.SOPInstanceUID    if hasattr(dcm,"SOPInstanceUID")    else "not present"
    inr =dcm.InstanceNumber    if hasattr(dcm,"InstanceNumber")    else "not present"
    print("\t\tseries UID: {}, series: {}, acquisition: {}, instance UID: {}, instance: {}".format(suid,sn,an,iuid,inr))

def read_scan(filelist,filedir):
    # filelist is a list of paths belonging to a single scan
    # the files are read and a volume is composed of them
    
    instances={}
    siuid=None
    
    for a in filelist:
        
        try:
            pdcm=dcmread(join(filedir,a))
        except InvalidDicomError:
            print("Could not read {}.".format(join(filedir,a)))
            continue
        
        # check if all files have the same SeriesInstanceUID
        seriuid=pdcm.SeriesInstanceUID
        if siuid is None:
            siuid=seriuid
        else:
            assert seriuid==siuid
            
        if not hasattr(pdcm,"ImagePositionPatient"):
            print("file {} has no ImagePositionPatient attribute".format(join(filedir,a)))
            continue
            
        ins=pdcm.SOPInstanceUID
        
        if ins in instances:
            report_repeated(instances[ins],pdcm)
            continue
            
        instances[ins]=pdcm

    oins=orderInstances([v for k,v in instances.items()])

    nv,src2xyz,xyz2src=getTransformedVolume(oins)
    
    voxel_spacing=getVoxelSpacingXYZ(oins,xyz2src)
    
    return nv,voxel_spacing

import SimpleITK as sitk
from lungmask import mask
from skimage.measure import label

def preprocess(volume):
    horizontal_volume = np.rot90(volume, k=-1)
    depth_first_volume = np.transpose(horizontal_volume, axes=(2, 0, 1))
    return depth_first_volume

def un_preprocess(volume):
    print(volume.shape)
    volume = np.transpose(volume,axes=(1,2,0))
    volume = np.rot90(volume,k=1)
    return volume

def extract_lungs(volume, segmentation):
    clipped = np.clip(volume, -1024,  600)
    masked = clipped.copy()
    masked[segmentation == 0] = -1024
    return masked

def normalize(volume):
    return (volume + 1024) / 812 - 1

def min_max_slice(indices):
    return slice(indices.min(), indices.max() + 1)

def crop(volume, segmentation):
    d_idx, h_idx, w_idx = np.nonzero(segmentation > 0)
    return volume[min_max_slice(d_idx), min_max_slice(h_idx), min_max_slice(w_idx)]

def removeSmallSegments(segmentation):
    segmentation[segmentation>0]=1
    l=label(segmentation)
    binedges=np.arange(100)
    hist,_=np.histogram(l,bins=binedges)
    ratios=hist[2:]/np.maximum(hist[1:-1],np.ones(1)).astype(float)
    last_label=np.argmax(ratios<0.3)+1
    segmentation[l>last_label]=0
    return segmentation

def segment_lungs(scan):

    preprocessed_volume = preprocess(scan)

    image = sitk.GetImageFromArray(preprocessed_volume)
    lung_mask = mask.apply(image)

    segmentation = removeSmallSegments(lung_mask)
    segmentation = un_preprocess(segmentation)

    lungs = extract_lungs(scan, segmentation)

    normalized = normalize(lungs)
    cropped = crop(normalized, segmentation)

    return cropped

p=argparse.ArgumentParser(description=
    "a quick and dirty script to preprocess CT scans")

req_args=p.add_argument_group("required arguments")
req_args.add_argument("--infiles",  type=str, required=True, nargs="+", 
    dest="infiles", help="a list of the input dicom file paths")
req_args.add_argument("--outfile", type=str, required=True, dest="outfile", 
    help="the output file path")

p.add_argument("--root_dir", type=str,
    help=("the root directory for output scan storage; if using --scan_db, "
          "split the path of the output file into two parts: "
          "the root_dir and the local_path; "
          "pass the root_dir to --root_dir and local_path to --outfile; "
          "the part passed to --outfile is inserted to the scan data base; "
          "the root_dir is used to write the file but not inserted in the db"))

p.add_argument("--scan_db", type=str,
    help=("path to the json file containing the scan data base to update; "
          "if this argument is specified, the scan db will be updated with"
          " patient name, scanner type, and scan date; "
          " patient name has to be specified with --patient;"
          " scanner type and scan date are extractated from "
          "the dicom input files; options --scanner and --date "
          "can be used to manually specify them instead"))
p.add_argument("--patient", type=str,
    help=("the patient identifier or name to put in the scan data base; "
          "if --scan_db is used, this argument must be specified "))
p.add_argument("--scanner", type=str,
    help=("the scanner brand and type to put in the scan data base; "
          "if --scan_db is used but this argument is left unspecified "
          "the scanner brand and type are extracted from the dicom files"))
p.add_argument("--date", type=date.fromisoformat,
    help="the scan date to put in the scan data base, in format: YYYY-MM-DD")
p.add_argument("--overwrite", action="store_true",
    help="if the entry in the scan data base exists, overwrite it")

if __name__=="__main__":
    a=p.parse_args()
    
    targetvoxelspacing=np.array([0.5,0.5]) # pixel size in mm
    
    outfn=a.outfile
    if a.root_dir is not None:
        outfn=join(a.root_dir,outfn)
    if isfile(outfn):
        print("File exists: {}".format(outfn))
        exit(2)
    
    dcm=dcmread(a.infiles[0],stop_before_pixels=True)
    dcm.noImgs=len(a.infiles)
    
    if preselect(dcm):                         
                            
        print("reading the scan")
        volume,voxel_spacing=read_scan(a.infiles,"./")
        print("computing the lung mask")
        try:
            masked=segment_lungs(volume)
        except ValueError:
            print("failed to mask the volume read from file {}"\
                  .format(a.infiles[0]))
            exit(3)
                            
        rv=rescaleSlices(masked,voxel_spacing[0:2],targetvoxelspacing)
        rv=np.ascontiguousarray(rv.transpose(2,1,0))
                            
        print("saving the scan to {}".format(outfn))
        np.save(outfn,rv)
    
        if a.scan_db is not None: # update the scan data base
            if isfile(a.scan_db):
                with open(a.scan_db,"r") as f:
                    scan_db=json.load(f)
                    f.close()
                print("updating the scan data base {}".format(a.scan_db))
            else:
                scan_db={}
                print("creating a new scan data base {}".format(a.scan_db))
            if a.outfile in scan_db and not a.overwrite:
                print("the entry {} exists in the scan data base {}"\
                      .format(a.outfile,a.scan_db))
                print("but --overwrite was not specified;")
                print("not updating the data base")
                exit(5)
            
            if a.patient is None:
                print("error: specified --scan_db but not --patient")
                print(("to update the scan data base {}"
                       " please specify patient name with --patient")\
                       .format(a.scan_db))
                exit(6)
            patient=a.patient 
            print("patient: {}".format(a.patient))
    
            if a.date:
                scan_date={"year" :a.date.year,
                           "month":a.date.month,
                           "day"  :a.date.day}
            else:
                scan_date={"year" :int(dcm.AcquisitionDate[:4]),
                           "month":int(dcm.AcquisitionDate[4:6]),
                           "day"  :int(dcm.AcquisitionDate[6:])}
            print("date: {}-{}-{}"\
                  .format(scan_date["year"],
                          scan_date["month"],
                          scan_date["day"]))
    
            scanner=a.scanner if a.scanner else \
                    dcm.Manufacturer+" "+dcm.ManufacturerModelName
            print("scanner: {}".format(scanner))
    
            scan_db[a.outfile]={"date"   :scan_date,
                                "patient":patient,
                                "scanner":scanner}
    
            print("writing the scan data base to {}".format(a.scan_db))
            with open(a.scan_db,"w") as f:
                json.dump(scan_db,f,sort_keys=True,indent=4)
                f.close()
            print("done")
    
    else:
        print("The scan does not satisfy initial selection criteria")
        exit(4)
