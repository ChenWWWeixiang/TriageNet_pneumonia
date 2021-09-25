import os,xlrd,json
import pydicom,glob
import pandas as pd
from scipy import stats
import numpy as np 
from xlrd import xldate_as_tuple
from datetime import datetime
datapath='/mnt/newdisk3/raw_data/all'
alldir_raw=glob.glob(datapath+'/*/*/')
alldir=[item.split('/')[-2] for item in alldir_raw]

train_list='for_dbz_pre/jsons/train.json'
eval_list='for_dbz_pre/jsons/val.json'
train_list=json.load(open(train_list,'r'))
eval_list=json.load(open(eval_list,'r'))
train_list = {**train_list, **eval_list}

allcls=['CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma']

agealllists=[[] for _ in range(17)]
genderlists=[[] for _ in range(17)]
sourcelists=[[] for _ in range(17)]


for item in train_list:
    if train_list[item]['clsII']=='healthy':
        continue
    clsidx=allcls.index(train_list[item]['clsII'])
    if isinstance(train_list[item]['age'],str):
        age=int(train_list[item]['age'][:-1])
    else:
        age=train_list[item]['age']
    genderlists[clsidx].append(train_list[item]['gender'])
    agealllists[clsidx].append(age)
    genderlists[-1].append(train_list[item]['gender'])
    agealllists[-1].append(age)
    if train_list[item]['source']=='检验科':
        ss=1
    else:
        ss=train_list[item]['source']
    sourcelists[clsidx].append(ss)


# for item in agealllists:
#     temp=np.array(item)
#     num1=np.sum(temp<=20)
#     num2=np.sum((20<temp) * (temp<=40))
#     num3=np.sum((40<temp) * (temp<=60))
#     num4=np.sum(60<temp )
#     print(num1,num2,num3,num4)
# for item in genderlists:
#     temp=np.array(item)
#     num1=np.sum(temp=="M")
#     num2=np.sum(temp=='F')
#     print(num1,num2)
# for item in sourcelists:
#     temp=np.array(item)
#     num0=np.sum(temp==0)
#     num1=np.sum(temp==1)
#     num2=np.sum(temp==2)
#     print(num0,num1,num2)
# a=1
sp=[]
for item in agealllists:
    a=np.percentile(np.array(item), (25, 50, 75), interpolation='midpoint')
    print('{:n}({:n}~{:n}),'.format(a[1],a[0],a[2]))
    for jtem in agealllists:
        if item==jtem:
            continue
        tempa=np.array(item)
        tempb=np.array(jtem)
        t, p = stats.ttest_ind(tempa,tempb)
        sp.append(p)
print(np.mean(sp))


sp=[]
for item in genderlists:
    r=np.sum(np.array(item)=='M')
    f=len(item)-np.sum(np.array(item)=='M')
    #a=np.percentile(r, (25, 50, 75), interpolation='midpoint')
    print('{:2}/{:2}'.format(r,f))
    for jtem in genderlists:
        if item==jtem:
            continue
        tempa=np.array(item)=='M'
        tempb=np.array(jtem)=='M'
        t, p = stats.ttest_ind(tempa,tempb)
        sp.append(p)
print(np.mean(sp))

print('GO!')
#allinfos=[]
allinfos=[[] for _ in range(17)]
train_list=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/c_all.json','r'))
f=open('temp.csv','w')
for item in train_list:
    if train_list[item]['clsII']=='healthy':
        continue
    thisone=train_list[item]
    #keys.append(item)
    clsidx=allcls.index(train_list[item]['clsII'])
    ss=thisone['basic_ill']+thisone['basic_sick']+thisone['blood_normal']+thisone['blood_2']

    allinfos[clsidx].append(ss)
    allinfos[-1].append(ss)
#allinfos=np.array(allinfos)

for feature in range(41):
    sp=[]
    temp=''
    for item in allinfos:
        if len(item)==0:
            #print('nan')
            continue
        tt=np.array(item)
        tt=tt[:,feature]
        nans=np.isnan(tt)
        tempa=tt[~nans]
        if len(tempa)==0:
            #print('nan')
            continue
        if feature>9:
            a=np.percentile(tempa, (25, 50, 75), interpolation='midpoint')
            temp+='{:n}({:n}~{:n})({:n}/{:n}),'.format(a[1],a[0],a[2],np.sum(~nans),len(nans))
        else:
            a=1
            temp+='{:n}/{:n}({:n}/{:n}),'.format(np.sum(tempa==0),np.sum(tempa==1),np.sum(~nans),len(nans))
        for jtem in allinfos:
            if len(jtem)==0:
                continue
            if item==jtem:
                continue
            tempb=np.array(jtem)
            mm=tempb[:,feature]
            nans=np.isnan(mm)
            tempb=mm[~nans]
            t, p = stats.ttest_ind(tempa,tempb)
            if not np.isnan(p):    
                sp.append(p)
    f.writelines(temp+'\n')
    print(np.mean(sp))
    a=1