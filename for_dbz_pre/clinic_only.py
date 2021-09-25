import pandas as pd
import numpy as np
import torch,json
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
from xgboost import plot_importance
from sklearn.metrics import accuracy_score  # 准确率
# 卡方检验，作为SelectKBest的参数
maintype=['healthy','virus','fungus','bacteria','chlamydia','mycoplasma',]
allsubtype= ['healthy','CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.model_selection import train_test_split
data=json.load(open('for_dbz_pre/jsons/clinical_only8.json','r'))



def ppa(df):
    df.pop('used_clinical_feature_idx')
    y = np.array([maintype.index(df[a]['clsI']) for a in df])
    y2 = np.array([allsubtype.index(df[a]['clsII']) for a in df])
    name =[a for a in df]
    X =  np.array([df[a]['clinic_f'] for a in df])
    return X,y,y2,name
def clsgogog(X_train,y_train,X_test, y_test):
    #62.6/44.8
    clf = make_pipeline(
        VarianceThreshold(threshold=1e-3),
        StandardScaler(), 
        #SVC(kernel='poly'),
        RandomForestClassifier()
        )
    clf.fit(X_train, y_train)
    
    print('Classification accuracy without selecting features: {:.3f}'
        .format(clf.score(X_test, y_test)))

method='xg'

def xgboost_go(X_train,y_train,X_test, y_test):
    # 65.51/46.51
    # 训练模型
    model = xgb.XGBClassifier(max_depth=25,learning_rate=0.1,n_estimators=5000,gama=1.2,silent=0,objective='multi:softmax')
    model.fit(X_train,y_train)
    
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print('accuarcy:%.2f%%'%(accuracy*100))
    
    # 显示重要特征
    plot_importance(model)
    plt.savefig('t.jpg')
def nn_go(X_train,y_train,X_test, y_test,clsnum):
    #62.6/44.2
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    model=torch.nn.Sequential(
        torch.nn.BatchNorm1d(X_train.shape[1]),
        torch.nn.Linear(X_train.shape[1],128),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(128,clsnum))
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,amsgrad=True,weight_decay=0.005)
    X_train,y_train,X_test, y_test=torch.Tensor(X_train).float().cuda(),torch.Tensor(y_train).long().cuda(),\
                                    torch.Tensor(X_test).float().cuda(),torch.Tensor(y_test).long().cuda()
    cri=torch.nn.NLLLoss().cuda()#0.3,0.7
    model=model.cuda()
    for i in range(5000):
        optimizer.zero_grad()
        pred=model(X_train)
        loss=cri(pred.log_softmax(-1),y_train)
        if i % 100==0:
            #print(loss.item())
            pred_np=torch.argmax(pred,-1).detach().cpu().numpy()
            acc=(pred_np==y_train.cpu().numpy()).astype(float).mean()
            #print(f'train acc at {i} step is {acc}!')
        loss.backward()
        optimizer.step()
    pred=model(X_test)
    pred_np=torch.argmax(pred,-1).detach().cpu().numpy()
    acc=(pred_np==y_test.cpu().numpy()).astype(float).mean()
    print('test acc',acc)
X_train, y_train,y_train2,name=ppa(data)

X_train,X_test,y_train,y_test,y_train2, y_test2 = train_test_split(X_train, y_train,y_train2, test_size=0.2, random_state=42)
#X_test, y_test,y_test2,name2=ppa(dft)
if method=='xb':
    xgboost_go(X_train,y_train,X_test, y_test)
    xgboost_go(X_train,y_train2,X_test, y_test2)
elif method=='nn':
    nn_go(X_train,y_train,X_test, y_test,6)
    nn_go(X_train,y_train2,X_test, y_test2,17)
else:
    clsgogog(X_train,y_train,X_test, y_test)
    clsgogog(X_train,y_train2,X_test, y_test2)

