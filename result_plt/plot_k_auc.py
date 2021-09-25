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
    if xx:
        res.append(np.mean(value))
    else:
        res.append(str(np.mean(value)) + ' (' + str(confidence_lower) + '-' + str(confidence_upper) + ')')
    return res
import argparse
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../NO5.npy','../NO2.npy','../top1.npy','../top3.npy','../top5.npy','../top9.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='csvs/kk.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
CLS=['Healthy','CAP','A/B Influenza','COVOD-19']
plt.figure()
axes = plt.subplot(111)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Distinguishing COVID from CAP')
axins = zoomed_inset_axes(axes, 2, loc="lower right",borderpad=2)  # zoom = 6
axins.set_xlim(0, 0.3)
axins.set_ylim(0.7, 1)


with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name', 'AUC','Accuracy'])
    for a_res in ress:
        res = np.load(a_res)
        if res.shape[1]==4:
            pre=np.array(res[:,:-1],np.float)
            gt=np.array(res[:,-1],np.float)
        else:
            pre = np.array(res[:, 1:-1], np.float)
            gt = np.array(res[:, -1], np.float)
        #AUC=[]
        #pre=pre/pre.sum(1,keepdims=True)
        ACC=[]
        REC=[]
        SPE=[]
        SAUC=[]
        y_one_hot = label_binarize(gt, np.arange(4))
        norm_x=pre/ pre.max(axis=0)
        AUC=[]
        for i in range(100):
            train_x, test_x, train_y, test_y = train_test_split(pre, y_one_hot, test_size=0.2)
            train_x=train_x/train_x.max(axis=0)
            auc = metric.roc_auc_score(train_y, train_x)
            AUC.append(auc)

            prediction = np.argmax(train_x, 1)
            groundtruth = np.argmax(train_y, 1)

            ACC.append(np.mean(prediction == groundtruth))

        Res=[a_res]
        Res = get_CI(AUC,Res)
        Res = get_CI(ACC, Res)
        f.writerow(Res)
       # plt.figure(1)

        fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())
        axes.plot(fpr, tpr, label=a_res+', AUC={:.2f}'.format(np.mean(AUC)))
        axins.plot(fpr, tpr)

axes.legend(loc="upper right")
plt.tight_layout()


plt.savefig('jpgs/kk.jpg')
