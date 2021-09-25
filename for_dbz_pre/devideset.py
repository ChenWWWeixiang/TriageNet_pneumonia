import json,random
import numpy as np
import sys
sys.path.append('/mnt/data9/Lipreading-DenseNet3D-master')
from for_dbz_pre.settings2d import allsubtype,maintype
new=False
train_list='for_dbz_pre/jsons/all_data.json'
train_list2='for_dbz_pre/jsons/more_healthy.json'

train_list3='for_dbz_pre/jsons/more.json'

train_list=json.load(open(train_list,'r'))
train_list2=json.load(open(train_list2,'r'))
train_list3=json.load(open(train_list3,'r'))
alls = {**train_list, **train_list2,**train_list3}
alls=json.load(open('for_dbz_pre/jsons/clinical_only8.json','r'))
allperson=list(alls.keys())
allperson=[p.split('.')[0] for p in allperson]
allperson=list(set(allperson))

traindict=dict()
valdict=dict()
type1 = np.zeros(6)
type2 = np.zeros(17)


train_old=json.load(open('for_dbz_pre/jsons/train.json','r'))
test_old=json.load(open('for_dbz_pre/jsons/val.json','r'))
if new:
    for item in allperson:
        thisone = alls[item]
        type1[maintype.index(thisone['clsI'])] += 1
        type2[allsubtype.index(thisone['clsII'])] += 1
    #c1 = np.zeros(5)
    c2 = np.zeros(17)
    random.shuffle(allperson)
    for item in allperson:
        thisone = alls[item]
        c2[allsubtype.index(thisone['clsII'])] += 1
        if c2[allsubtype.index(thisone['clsII'])]<type2[allsubtype.index(thisone['clsII'])]//2:
            traindict[item]=thisone
            if item+'.1' in alls.keys():
                traindict[item+'.1' ] = alls[item+'.1' ]
        else:
            valdict[item] = thisone
            if item + '.1' in alls.keys():
                valdict[item + '.1'] = alls[item + '.1']
else:
    for item in allperson:
        thisone = alls[item]
        if item in test_old.keys():
            valdict[item]=thisone
        else:
            traindict[item]=thisone

json.dump(traindict,open('for_dbz_pre/jsons/train_c8.json','w'))
json.dump(valdict,open('for_dbz_pre/jsons/val_c8.json','w'))
