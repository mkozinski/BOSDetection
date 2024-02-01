from random import shuffle
import json
import argparse
from os.path import isfile

def ntest_from_nsplits(n,nsplits):
    ntest=[]
    while n>0:
        ntest.append(round(n/nsplits))
        n-=ntest[-1]
        nsplits-=1
        
    return ntest 
        

def split_patient_list(patient_list, nsplits):
    
    shuffle(patient_list)

    split_map={}
    pat_ind=0
    spl_ind=0
    for n in nsplits:
        for i in range(n):
            split_map[patient_list[pat_ind]]=spl_ind
            pat_ind+=1
        spl_ind+=1

    return(split_map)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("patient_db",type=str)
    p.add_argument("nsplits",type=int)
    p.add_argument("output_file",type=str)
    
    a=p.parse_args()
    
    if isfile(a.output_file):
        print("{}: file {} exists".format(p.prog,a.output_file))
        exit(2)

    with open(a.patient_db,"r") as f:
        patient_db=json.load(f)

    patients_pos=[pat for pat in patient_db if patient_db[pat]["label"]==1]
    patients_neg=[pat for pat in patient_db if patient_db[pat]["label"]==0]

    n_test_pos=ntest_from_nsplits(len(patients_pos),a.nsplits)
    split_map=split_patient_list(patients_pos,n_test_pos)
    n_test_neg=ntest_from_nsplits(len(patients_neg),a.nsplits)
    split_map.update(split_patient_list(patients_neg,n_test_neg))

    with open(a.output_file,"w") as f:
        json.dump(split_map,f,sort_keys=True,indent=4)
        f.close()
