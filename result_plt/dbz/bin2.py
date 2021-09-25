
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
                    #'/mnt/data9/Lipreading-DenseNet3D-master/re/pure_forest.npy',
                    '/mnt/data9/Lipreading-DenseNet3D-master/re/pure_forest.npylevel3.npy',
                    
                    #'/mnt/data9/Lipreading-DenseNet3D-master/re/no_feature_select_groups.npylevel2.npy',
                    '/mnt/data9/Lipreading-DenseNet3D-master/re/resnet.npylevel2.npy',
                    #'/mnt/data9/Lipreading-DenseNet3D-master/re/temp_mild.npylevel3.npy',
                    # '/mnt/data9/Lipreading-DenseNet3D-master/re/onlyd2.npy',
                    # '/mnt/data9/Lipreading-DenseNet3D-master/re/resnet_onlyd.npy',
                    # '/mnt/data9/Lipreading-DenseNet3D-master/re/mlp2.npy',
                    ])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/level2.2.csv')
args = parser.parse_args()
f,ax=plt.subplots(figsize=(8,6))
#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
CLS=['virus','fungus','bacteria','chlamydia','mycoplasma']
CLS= ['CMV', 'Respiratory syncytial','COVID-19']+\
    ['Aspergillus', 'Candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['Chlamydia','Mycoplasma',]
TT=['TriageNet','ResNet','num of cases(k)']
all_record=[]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.axhline(y=0.5248, color=colors[0], linestyle='-')
#plt.axhline(y=0.4807, color=colors[1], linestyle='-')
plt.axhline(y=0.2448, color=colors[1], linestyle='-')
nums=[119,43,847,196,179,199,147,150,208,169,47,49]
idxx=np.argsort(-np.array(nums))
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name', 'AUC','Accuracy'])
    x = list(range(12))
    for ii,a_res in enumerate(ress):
        res = np.load(a_res)
        pred = np.array(res[:, -13:-1], np.float)
        gt = np.array(res[:, -1], np.float)
        pre=np.argmax(pred,-1)
        pre2=np.argsort(-pred,-1)[:,:2]
        y_one_hot = label_binarize(gt, np.arange(12))
        sen=[]
        spe=[]
        sauc=[]
        for cls in range(12):
            ss=0
            sen.append(np.sum((pre == cls) * (gt == cls)) / np.sum(gt == cls))
            spe.append(np.sum((pre != cls) * (gt != cls)) / np.sum(gt != cls))
            sauc.append(metric.roc_auc_score(y_one_hot[:,cls], pred[:,cls]))
        sen=np.array(sen)[idxx]
        w=0.2
        x=[item+w for item in x]
        if ii==1:
            plt.bar(x, sen,width=w,label=TT[ii],tick_label = np.array(CLS)[idxx].tolist())
        else:
            plt.bar(x, sen,width=w,label=TT[ii])
    x=[item+w for item in x]
    plt.bar(x, np.array(nums)[idxx]/1000,width=w,label=TT[-1])

#plt.figure(1)
#plt.xlabel('Level-I catogeries')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.ylabel('Accuracy')
#plt.title('Accuracy on different level-II pathogens for small-lesion group')
plt.legend()


plt.tight_layout()
plt.savefig('/mnt/data9/Lipreading-DenseNet3D-master/result_plt/dbz/levelII.2.jpg')

