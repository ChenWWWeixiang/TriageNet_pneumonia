import pandas as pd
import numpy as np
import torch,json
import torch.nn as nn
# 特征最影响结果的K个特征
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#from feature_selector import FeatureSelector
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn import linear_model
import matplotlib.pyplot  as plt
import xgboost as xgb
from merge_features import merge_all_features
from xgboost import plot_importance
from sklearn.metrics import accuracy_score,confusion_matrix  # 准确率
# 卡方检验，作为SelectKBest的参数
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
maintype=['healthy','virus','fungus','bacteria','chlamydia','mycoplasma',]
allsubtype= ['healthy','CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
from sklearn.feature_selection import mutual_info_classif,f_classif

f1_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train_multi.csv"
f2_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train.csv"
clinic='for_dbz_pre/jsons/clinical_only8.json'
dfs=['/mnt/data9/Lipreading-DenseNet3D-master/saves/X_dbz_t.npy','/mnt/data9/Lipreading-DenseNet3D-master/saves/Y_dbz_t.npy',
    '/mnt/data9/Lipreading-DenseNet3D-master/saves/Z_dbz_t.npy']
mod='d'
X_train, y_train,y_train2,name=merge_all_features(f2_path,f1_path,clinic,dfs,mod=mod)
f1_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_test_multi.csv"
f2_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/result_test.csv"
#cp='for_dbz_pre/jsons/clinical_only8.json'
dfs=['/mnt/data9/Lipreading-DenseNet3D-master/saves/X_dbz.npy','/mnt/data9/Lipreading-DenseNet3D-master/saves/Y_dbz.npy',
    '/mnt/data9/Lipreading-DenseNet3D-master/saves/Z_dbz.npy']
X_test, y_test,y_test2,name2=merge_all_features(f2_path,f1_path,clinic,dfs,mod=mod)


class Mixup(nn.Module):
    def __init__(self):
        super(Mixup,self).__init__()
        self.alpha=0.5
    def forward(self,x,y):
        mixed_x, y_a, y_b, lam=self.mixup_data(x,y)
        return  mixed_x, y_a, y_b, lam
    def mixup_data(self,x,y):
        lam = np.random.beta(self.alpha, 1-self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index] 
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion,pred,y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

mixor=Mixup()
def ppac(df):
    df.pop('used_clinical_feature_idx')
    y = np.array([maintype.index(df[a]['clsI']) for a in df])
    y2 = np.array([allsubtype.index(df[a]['clsII']) for a in df])
    name =np.array([df[a]['name'] for a in df])
    X =  np.array([df[a]['clinic_f'] for a in df])
    return X,y,y2,name

#%%%%V2.0
#xgboost:
#d 56.75%/22.70%
#r 59.14%/28.33%
#c 53.98%/26.42%
#drc 59.80%/33.81%
#dr 59.25%/27.24%
#rc 60.28%/35.89%  *
#dc 
#%%%%%V1.0
##using xgboost:
#base 58.50%/27.04%
#base+clinic 61.31%/36.28%
#clinic: 52.07%/ 27.58%

##using nn:
#base : 51.5
#base+clinic 
#clinic : 51.5%/ 18.74% (aug) 52.7% / 18.07%

def clsgogog(X_train,y_train,X_test, y_test):
    #62.6/44.8
    clf = make_pipeline(
        VarianceThreshold(threshold=1e-1),
        StandardScaler(), 
        #SVC(kernel='poly'),
        RandomForestClassifier()
        )
    clf.fit(X_train, y_train)
    
    print('Classification accuracy without selecting features: {:.3f}'
        .format(clf.score(X_test, y_test)))
    clf = LinearSVC()
    clf_selected = make_pipeline(
            VarianceThreshold(threshold=1e-1),
           
            StandardScaler(),
            SelectKBest(mutual_info_classif, k=100),
            #RFE(clf,n_features_to_select=8),
            #SVC(kernel='poly'),
            RandomForestClassifier()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after univariate feature selection: {:.3f}'
        .format(clf_selected.score(X_test, y_test)))
method='nn'

def xgboost_go(X_train,y_train,X_test, y_test):
    # 65.51/46.51
    # 训练模型
    model = xgb.XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=200,silent=1,objective='multi:softmax')
    
    
    model.fit(X_train,y_train)
    #softprob_pred = model.predict(X_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)  
    
    # 计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print('accuarcy:%.2f%%'%(accuracy*100))
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    # 显示重要特征
    plot_importance(model)
    plt.savefig('t.jpg')
def every_acc(pred,gt,num_cls):
    acc=np.zeros(num_cls)
    for i in num_cls:
        acc[i]=((pred==gt).astype(float)*(gt==i).astype(float)).mean()
    return acc

def nn_go(X_train,y_train,X_test, y_test,clsnum):
    #62.6/44.2
    X_train=np.array(X_train).astype(np.float)
    X_test=np.array(X_test).astype(np.float)
    X_train=X_train/X_train.std(0)[np.newaxis,:].repeat(y_train.shape[0],0)
   # X_test=np.array(X_test)
    X_test=X_test/X_test.std(0)[np.newaxis,:].repeat(y_test.shape[0],0)
    y_test=np.array(y_test)
    model=torch.nn.Sequential(
        torch.nn.BatchNorm1d(X_train.shape[1]),
        torch.nn.Linear(2048,1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        # torch.nn.Linear(1024,1024),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(),
        torch.nn.Linear(1024,clsnum))
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001,amsgrad=True)
    X_train,y_train,X_test, y_test=torch.Tensor(X_train).float().cuda(),torch.Tensor(y_train).long().cuda(),\
                                    torch.Tensor(X_test).float().cuda(),torch.Tensor(y_test).long().cuda()
    cri=torch.nn.NLLLoss().cuda()#0.3,0.7
    model=model.cuda()
    mm=0
    for i in range(50000):    
        optimizer.zero_grad()
        pred=model(X_train)
        loss=cri(pred.log_softmax(-1),y_train)
        mixed_x, y_a, y_b, lam=mixor(X_train,y_train)
        mixed_p=model(mixed_x).log_softmax(-1)
        loss+=mixor.mixup_criterion(cri,mixed_p,y_a, y_b, lam)
        mm+=loss.item()
        if i % 1000==1:
            print(mm/1000)
            mm=0
            #pred_np=torch.argmax(pred,-1).detach().cpu().numpy()
            #acc=(pred_np==y_train.cpu().numpy()).astype(float).mean()
            #print(f'train acc at {i} step is {acc}!')
        loss.backward()
        optimizer.step()
    pred=model(X_test)
    pred_np=torch.argmax(pred,-1).detach().cpu().numpy()
    acc=(pred_np==y_test.cpu().numpy()).astype(float).mean()
    print('test acc',acc)

#X_train, y_train,y_train2,name=ppa(df)
#X_test, y_test,y_test2,name2=ppa(dft)
a=1
if method=='xb':
    xgboost_go(X_train,y_train,X_test, y_test)
    xgboost_go(X_train,y_train2,X_test, y_test2)
elif method=='nn':
    nn_go(X_train,y_train,X_test, y_test,5)
    nn_go(X_train,y_train2,X_test, y_test2,16)
else:
    clsgogog(X_train,y_train,X_test, y_test)
    clsgogog(X_train,y_train2,X_test, y_test2)

