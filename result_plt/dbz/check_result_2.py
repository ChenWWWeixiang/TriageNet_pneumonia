import xlrd,os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import sklearn.metrics as metric
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
refmap=[0,0,0,1,1,1,2,2,2,2,2,3,4]
res=np.load('/mnt/data9/Lipreading-DenseNet3D-master/re/reader2_new.npy')
pre = np.array(res[:, 1:-1], np.float)
gt = np.array(res[:, -1], np.float)
print(np.mean(np.argmax(pre,-1) == gt))
AUC=[]
#pre=pre/pre.sum(1,keepdims=True)
ACC=[]
REC=[]
SPE=[]
SAUC=[]
norm_x=pre
for i in range(100):
    train_x, test_x, train_y, test_y = train_test_split(norm_x, gt, test_size=0.2)
    auc = metric.roc_auc_score(train_y, train_x[:,1], average='micro')
    AUC.append(auc)

    prediction = np.argmax(train_x, 1)
    groundtruth = train_y

    ACC.append(np.mean(prediction == groundtruth))

Res=[res]
Res = get_CI(AUC,Res)
Res = get_CI(ACC, Res)
#print(Res)
# plt.figure(1)

fpr, tpr, thresholds = metric.roc_curve(gt, norm_x[:,1])
plt.plot(fpr, tpr, label='DFN, AUC={:.2f}'.format(np.mean(AUC)))
cm=confusion_matrix(gt,norm_x[:,1]>0.5,normalize='true')
print(cm)

answer='answer2.txt'
humans=[
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/7-reader record-1年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/5-reader record-2年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/1-reader record-5年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/2-reader record-6年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/3-reader record-12年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/6-reader record-12年.xlsx',
'/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/4-Reader record-32年.xlsx',
]

for onep in humans:
    with open(answer,'r') as f:
        aa=f.readlines()
        gt=['virus' in a.split(',')[1] for a in aa]
        data=pd.read_excel(onep,'测试2')
        data=np.array(data)[3:,:6].astype(np.float)
       
        data[np.isnan(data)]=0
        year=int(onep.split('年')[0].split('-')[-1])

        #pdb.set_trace()

        predmain=data[:,1]
        cm=confusion_matrix(gt,predmain,normalize='true')
        print(cm)
        predmain=np.array(predmain)
        gt=np.array(gt)
        specifciity = np.sum((gt == 0) * (predmain ==0)) / np.sum(gt == 0)
        sensitivity = np.sum((gt == 1) * (predmain ==1)) / np.sum(gt == 1)
       # ACC=pre==gt

        plt.scatter(1-specifciity,sensitivity)


plt.legend(['deep network','reader 1-y','reader 2-y','reader 5-y','reader 6-y','reader 12-y(a)','reader 12-y(b)',
            'reader 32-y'])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reader study II: Virus Recognition ROC Curves')
plt.savefig('/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/reader2.jpg',bbox_inches='tight')
