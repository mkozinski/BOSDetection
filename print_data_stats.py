import json

from datetime import date

import sys
import os
import argparse

p=argparse.ArgumentParser(description=(
    "show statistics of scans and patients "
    "described in the patient_db.json and scan_db.json files "
    "specify either the directory where the files are located "
    "with --db_dir, or the path to each of these files "
    "with --scan_db and --patient_db"))
p.add_argument("--db_dir",type=str,help=(
    "the directory containing the scan_db.json"
    " and the patient_db.json files; "
    "do not use together with --scan_db or --patient_db"))
p.add_argument("--scan_db",type=str,help=(
    "the path to the json file containing the scan data base"
    "do not use together with --db_dir"))
p.add_argument("--patient_db",type=str,help=(
    "the path to the json file containing the patient data base"
    "do not use together with --db_dir"))
a=p.parse_args()

errormsg=": specify either --db_dir or both --scan_db and --patient_db"
patient_db_fname="patient_database.json"
scan_db_fname   ="scan_database.json"
patient_db_path=None
scan_db_path   =None
if a.db_dir is not None:
    if a.scan_db is not None or a.patient_db is not None:
        print(p.prog+errormsg)
        sys.exit(1)
    patient_db_path=os.path.join(a.db_dir,patient_db_fname)
    scan_db_path   =os.path.join(a.db_dir,scan_db_fname)
elif a.scan_db is not None and a.patient_db is not None:
    patient_db_path=a.patient_db
    scan_db_path   =a.scan_db
else:
    print(p.prog+errormsg)
    sys.exit(2)

assert patient_db_path is not None
assert scan_db_path    is not None
    
with open(patient_db_path,"r") as f:
    patient_db=json.load(f)
    f.close()

with open(scan_db_path,"r") as f:
    scan_db=json.load(f)
    f.close()

d0=date(1900,1,31)
cond_nonBOS   =lambda p,s: patient_db[p]["label"]==0
cond_BOSbefore=lambda p,s: patient_db[p]["label"]==1 and \
                           (date(**patient_db[p]["FEV1_level_dates"]['0.8'])-date(**scan_db[s]["date"])).days> 0
cond_BOSafter =lambda p,s: patient_db[p]["label"]==1 and \
                           (date(**patient_db[p]["FEV1_level_dates"]['0.8'])-date(**scan_db[s]["date"])).days<=0
cond_BOSall   =lambda p,s: patient_db[p]["label"]==1

for name,cond in zip(["non-BOS patients","BOS after diag","BOS before diag","all BOS"],
                     [cond_nonBOS,       cond_BOSafter,    cond_BOSbefore  ,cond_BOSall]):
    pat_freq={}
    scanner_freq={}
    for s,scandict in scan_db.items():
        p=scandict["patient"]
        if cond(p,s):
            scandate=scan_db[s]["date"]
            scandate=date(scandate["year"],scandate["month"],scandate["day"])
            intdate=(scandate-d0).days
            if p not in pat_freq:
                pat_freq[p]={}
            pdict=pat_freq[p]
            if intdate not in pdict:
                pdict[intdate]=[]
            pdict[intdate].append(s)

            scanner=scan_db[s]["scanner"]
            if scanner not in scanner_freq:
                scanner_freq[scanner]=1
            else:
                scanner_freq[scanner]+=1
                
    print("{} ({} patients)".format(name,len(pat_freq)))
    total_visits=0
    total_scan_copies=0
    for p in pat_freq:

        nvisits=len(pat_freq[p])
        total_visits+=nvisits

        ncopies=len([s for d in pat_freq[p].values() for s in d])
        total_scan_copies+=ncopies

        print("\t{:7}\t{:4}\t{:4}".format(p,nvisits,ncopies))

    print("{}{}{}".format(" "*4,"total","-"*19))    
    print("\t{:7}\t{:4}" .format("visists"    ,total_visits))
    print("\t{:15}\t{:4}".format("scan copies",total_scan_copies))

    print("{}{}".format(" "*4,"scanners used"))
    for s in scanner_freq:
        print("\t{:40}{:4}".format(s,scanner_freq[s]))
    print()
    
