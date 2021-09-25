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
                    default=[
                   # '/mnt/data9/Lipreading-DenseNet3D-master/re/try.npylevel2.npy',
                    '/mnt/data9/Lipreading-DenseNet3D-master/re/pure_forest.npylevel3.npy',
                    '/mnt/data9/Lipreading-DenseNet3D-master/re/resnet.npylevel2.npy',
                    '/mnt/data9/Lipreading-DenseNet3D-master/re/no_feature_select_groups.npylevel2.npy',
                    #'/mnt/data9/Lipreading-DenseNet3D-master/re/no_feature_select.npylevel2.npy',
                    ])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/method_roc2.2.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
CLS=['virus','fungus','bacteria','chlamydia','mycoplasma']
#TT=['pos-aware DDAF','DDAF','resnet','resnet+groupsoft']
TT=['DDAF']
TT=['TriageNet','ResNet','Groupsoftmax']
all_record=[]
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name', 'AUC','Accuracy'])
    for ii,a_res in enumerate(ress):
        res = np.load(a_res)

        pre = np.array(res[:, 1:-1], np.float)
        gt = np.array(res[:, -1], np.float)
        AUC=[]
        #pre=pre/pre.sum(1,keepdims=True)
        ACC=[]
        REC=[]
        SPE=[]
        SAUC=[]

        y_one_hot = label_binarize(gt, np.arange(12))
        norm_x=pre
        for i in range(200):
            train_x, test_x, train_y, test_y = train_test_split(pre, y_one_hot, test_size=0.2)
            #train_x=train_x
            auc = metric.roc_auc_score(train_y, train_x, average='micro')
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
        plt.plot(fpr, tpr, label=TT[ii])


plt.figure(1)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Overall level-II ROC Curves')
plt.legend()
plt.savefig('/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/method_roc.2.jpg')

