import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train_multi.csv"
dft = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train_3.csv"
cp='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_only8.json'
allj='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/all_data.json'
dfs=['/mnt/data9/Lipreading-DenseNet3D-master/saves/X_dbz.npy','/mnt/data9/Lipreading-DenseNet3D-master/saves/Y_dbz.npy','/mnt/data9/Lipreading-DenseNet3D-master/saves/Z_dbz.npy']
maintype=['healthy','virus','fungus','bacteria','chlamydia','mycoplasma',]
allsubtype= ['healthy','CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
def merge_all_features(f1_path,f2_path,clinic,dfs,mod='cdr'):
    da1=pd.read_csv(f1_path)
    y1=da1.pop("label1")
    y2=da1.pop("label2")
    if f2_path:
        da2=pd.read_csv(f2_path)
        y1=da2.pop("label1")
        y2=da2.pop("label2")
        rdata=pd.concat([da1,da2])#1380,3968
    else:
        rdata=da1
    cdata=json.load(open(clinic,'r'))
    F=[]
    for c in cdata:
        if c=='used_clinical_feature_idx':
            continue
        F.append([cdata[c]['newpath'].split('/')[-1].split('-')[1]]+cdata[c]['clinic_f'])
    F=np.array(F)
    cdata=pd.DataFrame(F,columns=['Patient','f1','f2','f3','f4','f5','f6','f7','f8'])#1475,9
    #dd=pd.merge(dd,cdata,on='Patient')#726,3976
    # name=np.load(dfs[2],allow_pickle=True)
    # name=np.array([n[0].split('/')[-1].split('-')[1] for n in name])
    # name=name[:,np.newaxis]
    # XX=[]
    # X=np.load(dfs[0],allow_pickle=True)
    # for x in X:
    #     XX.append(np.mean(x,0))
    # X=np.array(XX)
    # dfdata=np.concatenate([name,X],1)
    # dfdata=pd.DataFrame(dfdata,columns=['Patient']+np.arange(0,2048).astype(np.str).tolist())#1291,2048
    

    if 'c' in mod:
        if 'd' in mod:
            dd=pd.merge(cdata,dfdata,on='Patient')
            if 'r' in mod:
                dd=pd.merge(dd,rdata,on='Patient')
        else:
            if 'r' in mod:
                dd=pd.merge(rdata,cdata,on='Patient')
            else:
                dd=pd.merge(cdata,dfdata,on='Patient').iloc[:,:9]
    else:
        if 'd' in mod:
            if 'r' in mod:
                dd=pd.merge(dfdata,rdata,on='Patient')
            else:
                dd=dfdata
        else:
            if 'r' in mod:
                dd=rdata
            else:
                dd=[]
    ad=json.load(open(allj,'r'))
    allname=[ad[item]['newpath'].split('/')[-1].split('-')[1] for item in ad]
    c1=[maintype.index(ad[item]['clsI']) for item in ad]
    c2=[allsubtype.index(ad[item]['clsII']) for item in ad]
    label1=[]
    label2=[]
    name=dd.pop('Patient').tolist()
    dd=np.array(dd)
    X=[]
    for ii,item in enumerate(name):
        if item in allname:
            idx=allname.index(item)
            label1.append(c1[idx])
            label2.append(c2[idx])
            X.append(dd[ii,:])
    #a=1
    
    return np.array(X),np.array(label1),np.array(label2), np.array(name)


import pickle
f1_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train.csv"
#f1_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/results_train.csv"
#clinic='for_dbz_pre/jsons/clinical_only8.json'
dfs=['/mnt/data9/Lipreading-DenseNet3D-master/saves/X_dbz_t.npy','/mnt/data9/Lipreading-DenseNet3D-master/saves/Y_dbz_t.npy',
    '/mnt/data9/Lipreading-DenseNet3D-master/saves/Z_dbz_t.npy']
#X=merge_all_features(f1_path,f2_path,clinic,dfs,mod='cr')
#pickle.dump(X,open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/train_rc.pkl','wb'))

f2_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/result_test.csv"
#f2_path = "/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/result_test.csv"
clinic='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_only8.json'
X_train, y_train,y_train2,_=merge_all_features(f1_path,f2_path,clinic,dfs,mod='r')
#pickle.dump(X,open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/all_r2.pkl','wb'))
import torch
import os,pdb
os.environ['CUDA_VISIBLE_DEVICES']='0'
X_train,X_test,y_train,y_test,y_train2, y_test2 = train_test_split(X_train, y_train,y_train2, test_size=0.5, random_state=42)
def nn_go(X_train,y_train,X_test, y_test,clsnum):
    #
    X_train=np.array(X_train)
    X_train[np.isnan(X_train)]=0
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    X_test[np.isnan(X_test)]=0
    y_test=np.array(y_test)
    model=torch.nn.Sequential(
        torch.nn.BatchNorm1d(X_train.shape[1]),
        torch.nn.Linear(X_train.shape[1],512),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(512,clsnum))
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
        model[0].weight.unsqueeze(0),
    # w=model[0].weight.unsqueeze(1).repeat(1,6)*torch.mm(model[1].weight.transpose(1,0),model[-1].weight.transpose(1,0))
    # name_c=pd.read_csv(f1_path)
    # name_c=name_c.columns[3:]
    # for i in range(1,6):
    #     coef = pd.Series(w[:,i].detach().cpu().numpy(), index = name_c)
    #     imp_coef = pd.concat([coef.sort_values().head(3),
    #                  coef.sort_values().tail(3)])
    #     imp_coef.plot(kind = "barh")
    #     #plt.figure()
    #     plt.title(f"Most important coefficients for {maintype[i]}")
    #     plt.savefig(f"for_dbz_pre/jpgs/coffs_{maintype[i]}.jpg", bbox_inches='tight')
    #     plt.close()
    
    pred=model(X_test)
    pred_np=torch.argmax(pred,-1).detach().cpu().numpy()
    tosvae=np.concatenate([y_test.cpu().numpy()[:,np.newaxis],
                    pred.softmax(-1).detach().cpu().numpy(),y_test.cpu().numpy()[:,np.newaxis]],1)
    np.save('/mnt/data9/Lipreading-DenseNet3D-master/re/mlp.npy',tosvae)
    acc=(pred_np==y_test.cpu().numpy()).astype(float).mean()
    print('test acc',acc)
    a=1
#nn_go(X_train,y_train2,X_test, y_test2,17)
nn_go(X_train,y_train,X_test, y_test,6)