from os import path,makedirs
from datetime import date,timedelta
import numpy as np
from random import randrange,random
import json

root_dir="./PatientData"
scan_db_fname   =path.join(root_dir,"scan_database.json")
patient_db_fname=path.join(root_dir,"patient_database.json")
split_map_fname =path.join(root_dir,"split_map.json")

makedirs(root_dir)

patient_db={}
scan_db   ={}
split_map ={}

npatients=10
nscans   =8
FEV1_levels=['start','0.9','0.8','0.65','0.5']
date_0=date(1990,1,31)
i_patient=0
i_scan   =0

def create_scan(i_scan,root_dir,d,patname,scan_db):
    scandirname="scan_{}_{}_{}".format(d.year,d.month,d.day)
    scandir=path.join(patname,scandirname)
    makedirs(path.join(root_dir,scandir))
    for i_copy in range(randrange(1,3)):
        scanname=path.join(scandir,"scan_{:0>4d}_copy_{:0>2d}.npy".format(i_scan,i_copy))
        scan=np.random.rand(30,512,384)
        np.save(path.join(root_dir,scanname),scan)
        scan_db[scanname]={ "patient":patname,
                            "date":{"year":d.year,
                                    "month":d.month,
                                    "day":d.day,},
                            "scanner":"Scanner A" if random()>0.5 else "Scanner B",
                          }

for lbl in [0,1]:
    for i in range(npatients):
        patname="patient_{:0>2d}".format(i_patient)
        patdir =path.join(root_dir,patname)
        makedirs(patdir)
        split_map[patname]=i_patient%2

        patdict={"label":lbl,}
      
        d=date_0+timedelta(days=randrange(183,365))
        if lbl==1:
            scans_created=0
            fev1_lvl_dict={}
            nscans_per_level,_=np.histogram(np.random.rand(nscans),
                                          len(FEV1_levels),
                                          (0.0,1.0))
            for f1lvl,ns in zip(FEV1_levels,nscans_per_level):
                fev1_lvl_dict[f1lvl]={"year" :d.year,
                                      "month":d.month,
                                      "day"  :d.day} 
                for scanno in range(ns):
                    create_scan(i_scan,root_dir,d,patname,scan_db)
                    scans_created+=1
                    i_scan+=1
                    d=d+timedelta(days=randrange(183,365))
                    
                if scans_created>=nscans:
                    break
            patdict["FEV1_level_dates"]=fev1_lvl_dict
        else: # lbl==0
            for scanno in range(nscans):
                create_scan(i_scan,root_dir,d,patname,scan_db)
                i_scan+=1
                d=d+timedelta(days=randrange(183,365))
                        
        i_patient+=1
        patient_db[patname]=patdict
             
with open(scan_db_fname,"w") as f:
    json.dump(scan_db,f,sort_keys=True,indent=4)
    f.close()
             
with open(patient_db_fname,"w") as f:
    json.dump(patient_db,f,sort_keys=True,indent=4)
    f.close()
             
with open(split_map_fname,"w") as f:
    json.dump(split_map,f,sort_keys=True,indent=4)
    f.close()

