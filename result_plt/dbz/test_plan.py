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
T=np.array([[0,0.4,0,0,0,0,0,0,0,0,0.3,0.3],
            [0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0.5,0.5],
            [0,1,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0.1,0.2,0.2,0.2,0.2,0.2,0.2,0,0],
            [0,0,0,0.5,0.5,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0]])
T=np.array([[0,1,0,0,0,0,0,0,0,0,1,1],
            [0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,1],
            [0,1,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0]])
#T=(np.exp(T).T/np.exp(T).sum(0).T).T #1.62
T=T/T.sum(1,keepdims=True)
T=T/T.sum(0,keepdims=True)
GT_map=[[3],[0,4],[1],[5,6,7],[5,6],[5],[5],[5],[5],[5],[0,1],[0,1]]
#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress

for ii,a_res in enumerate(ress):
    res = np.load(a_res)
    pre = np.array(res[:, 1:-1], np.float)
    gt = np.array(res[:, -1], np.float)
    pred=np.matmul(pre,T.T)
    gtt=[GT_map[int(item)] for item in gt.tolist()]
    acc=[]
    num=[]
    for i in range(pred.shape[0]):
        p=pred[i,:].argmax()
        g=gtt[i]
        acc.append(p in g)
        asr=np.argsort(-pred[i,:])
        minidx=100
        for case in g:
            idx=np.where(asr==case)[0]
            if idx<minidx:
                minidx=idx
        num.append(minidx+1)
    print(np.mean(acc))
    print(np.mean(num))
    a=1
 

