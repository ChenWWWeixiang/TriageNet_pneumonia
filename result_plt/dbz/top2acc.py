##plot 3cls roc
import seaborn as sns
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
def get_CI(value,res,xx=False):
    sorted_scores=np.array(value)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    dis=np.mean(confidence_upper-confidence_lower)/2
    if xx:
        res.append(np.mean(value))
    else:
        res.append(str(np.mean(value)) + '+-' + str(dis))
    return res
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['/mnt/data9/Lipreading-DenseNet3D-master/re/pure_forest.npylevel3.npy'])

args = parser.parse_args()
T=np.array([[0,0.25,0,0,0,0,0,0,0,0.4,0.4],
            [0,0,1,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0.6,0.6],
            [0,0.75,0,0,0,0,0,0,0,0,0],
            [0,0,0,0.1,0.25,1,1,1,1,0,0],
            [0,0,0,0.3,0.75,0,0,0,0,0,0],
            [0,0,0,0.6,0,0,0,0,0,0,0]])

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress

for ii,a_res in enumerate(ress):
    res = np.load(a_res)

    pre = np.array(res[:, -6:-1], np.float)
    gt = np.array(res[:, -1], np.float)
    #AUC=[]
    acc,cnt=0,0
    pre_idx=np.argsort(-pre,-1)[:,:2]
    for i,f in enumerate(gt):
        if f in pre_idx[i].tolist():
            acc+=1
        cnt+=1
        a=1
    print(acc/cnt)
 

