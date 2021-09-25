import pandas as pd
import numpy as np
import pdb
from sklearn.metrics import confusion_matrix 
from settings2d import TYPEMAT,maintype
humans=['for_dbz_pre/1-reader record-5年.xlsx',
'for_dbz_pre/2-reader record-6年.xlsx',
'for_dbz_pre/3-reader record-12年.xlsx',
'for_dbz_pre/4-Reader record-32年.xlsx',
'for_dbz_pre/5-reader record-2年.xlsx',
'for_dbz_pre/6-reader record-12年.xlsx',
'for_dbz_pre/7-reader record-1年.xlsx']
refmap=[0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,4]
answers=['answer1.txt','answer2.txt']
A1,A2,A3,Am=[],[],[],[]
for onep in humans:
    with open(answers[0],'r') as f:
        aa=f.readlines()
        gt=[int(a.split(',')[-1]) for a in aa]
        data=pd.read_excel(onep,'测试1')
        data=np.array(data)[3:,:6].astype(np.float)
       
        data[np.isnan(data)]=0
        year=int(onep.split('-')[-1].split('年')[0])
        acc1=((data[:,1]-1)==gt).mean()
        acc5=[(data[:,-ic]-1)==gt for ic in range(5)]
        acc5=np.stack(acc5,1)
        acc5=((acc5.sum(-1)*1.0)>0).mean()
        #pdb.set_trace()
        gtmain=[refmap[ag] for ag in gt]
        accmain=[refmap[int(ag)] for ag in (data[:,1]-1).tolist()]
        cm=confusion_matrix(gtmain,accmain,normalize='true')
        print(cm)
        accmain=(np.array(accmain)==np.array(gtmain)).mean()
        
        A1.append(acc1)
        A2.append(acc5)
        Am.append(accmain)
    with open(answers[1],'r') as f:
        #pdb.set_trace()
        aa=f.readlines()
        gt=[int(a.split(',')[1].split('/')[4]=='virus') for a in aa]
        data=pd.read_excel(onep,'测试2')
        data=np.array(data)[3:,:2]
        year=int(onep.split('-')[-1].split('年')[0])
        acc=((data[:,-1])==gt).mean()
        print(f"A {year}-year doctor got {acc1}/{acc5} for top-1/5 ClsII acc, and {accmain} CLsI acc, and  got {acc} acc in virus-non test.")
        A3.append(acc)
print(f'Mean of 7 doctor: acc1 {np.mean(A1)}  acc5 {np.mean(A2)} acc_main {np.mean(Am)} acc_virus {np.mean(A3)}')
