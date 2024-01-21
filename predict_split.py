import numpy as np
import torch
import os.path
import sys

import json

import argparse

sys.path.append("../")
from NetworkTraining_py.crop import crop

sys.path.append("./")
from net import MyNet
from predict import getInput, getNet, predict

p=argparse.ArgumentParser()
p.add_argument("data_dir",type=str)
p.add_argument("scan_db",type=str)
p.add_argument("split_map",type=str)
p.add_argument("split_num",type=int)
p.add_argument("netpath",type=str)
p.add_argument("outfile",type=str)
p.add_argument("--cuda",action="store_true")

a=p.parse_args()

if os.path.isfile(a.outfile):
    print("File exists: {}".format(a.outfile))
    sys.exit(2)

net=getNet(a.netpath,a.cuda) 

with open(a.scan_db,"r") as f:
    scan_db=json.load(f)
    f.close()

with open(a.split_map,"r") as f:
    split_map=json.load(f)
    f.close()

results={}
for f,attr in scan_db.items():
    pat=attr["patient"]
    if pat not in split_map:
        continue
    test_split=split_map[pat]
    if test_split!=a.split_num:
        continue
    
    inp=getInput(os.path.join(a.data_dir,f))
    res=predict(net,inp)
    print("{:73} {:>6.2f}".format(f,res.reshape(-1)[0].item()))
    results[f]=res

np.save(a.outfile,{"split":a.split_num,"prediction":results},allow_pickle=True)
