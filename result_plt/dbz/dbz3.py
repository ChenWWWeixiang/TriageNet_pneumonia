##plot 3cls roc
import seaborn as sns
import os
import csv,pdb
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
    if xx:
        res.append(np.mean(value))
    else:
        res.append(str(np.mean(value)) + '+-' + str((confidence_upper-confidence_lower)/2))
    return res
import argparse
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import  pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['/mnt/data9/Lipreading-DenseNet3D-master/re/only-virus.npylevel2.npy',
                    ])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/virus.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
#CLS=['virus','fungus','bacteria','chlamydia','mycoplasma']
allsubtype= ['CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']
plt.figure()
axes = plt.subplot(111)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Distinguishing level-II virus using DFN')

all_record=[]
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name', 'AUC','Sensitivity','Specificity'])
    for a_res in ress:
        res = np.load(a_res)
        pre = np.array(res[:, 1:-1], np.float)
        gt = np.array(res[:, -1], np.float)
       # pdb.set_trace()
        #AUC=[]
        #pre=pre/pre.sum(1,keepdims=True)
        ACC=[]
        REC=[]
        SPE=[]
        SAUC=[]
        AUC=[]
        y_one_hot = label_binarize(gt, np.arange(5))
        pre=pre[:,:5]
        for i in range(100):
            train_x, test_x, train_y, test_y = train_test_split(pre, y_one_hot, test_size=0.2)
            #train_x=train_x/train_x.max(axis=0)

            auc = metric.roc_auc_score(train_y, train_x, average='micro')

            AUC.append(auc)

            prediction = np.argmax(train_x, 1)
            groundtruth = np.argmax(train_y, 1)

            ACC.append(np.mean(prediction == groundtruth))
            sen = []
            spe = []
            sauc = []
            for cls in range(5):
                sen.append(np.sum((prediction == cls) * (groundtruth == cls)) / np.sum(groundtruth == cls))
                spe.append(np.sum((prediction != cls) * (groundtruth != cls)) / np.sum(groundtruth != cls))
                sauc.append(metric.roc_auc_score(train_y[:,cls], train_x[:,cls]))
            SAUC.append(sauc)
            REC.append(sen)
            SPE.append(spe)
        SPE = np.array(SPE)
        REC = np.array(REC)
        SAUC = np.array(SAUC)
        for cls in range(5):
            Res=[allsubtype[cls]]
            Res = get_CI(SAUC[:, cls], Res)
            Res = get_CI(REC[:, cls], Res)
            Res = get_CI(SPE[:, cls], Res)
            f.writerow(Res)
            fpr, tpr, thresholds = metric.roc_curve(y_one_hot[:,cls], pre[:,cls])
            all_record.append(fpr)
            all_record.append(tpr)
            #axins.plot(fpr, tpr)
            axes.plot(fpr, tpr, label='{}, AUC={:.4f}'.format(allsubtype[cls],np.mean(SAUC[:, cls])))
        Res=['all']
        Res=get_CI(AUC,Res)
        Res = get_CI(ACC, Res)
        f.writerow(Res)

axes.legend()
plt.tight_layout()
plt.savefig('roc_virus.jpg')
