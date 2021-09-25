import json
import numpy as np
normal=[0,0,0,0,0,0,0,0,0,0,
    36.8,130,130,80,15,6,2,150,	230,5.1,75,	74,	43,	13,	38,47,0.3,13,	34,	17,	3,3,6,	18,	7.4,	90,
	40,24,5.2,156,0.4]
allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','healthy',]
data=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/c_all.json','r'))
allinfos=[]
keys=[]
for item in data.keys():
    thisone=data[item]
    if thisone['clsII'] not in allsubtype:
        continue
    keys.append(item)
    allinfos.append(thisone['basic_sick']+thisone['blood_normal'])
   # print(len(thisone['basic_ill']+thisone['basic_sick']+thisone['blood_normal']+thisone['blood_2']))
    
allinfos=np.array(allinfos)
pure=[]
print(allinfos.shape)
for i in range(allinfos.shape[0]):
    if np.isnan(allinfos[i,:]).any():
        continue
    else:
        #rowidx.append(i)
        pure.append(allinfos[i,:])
pure=np.array(pure)
means=pure.mean(0)

# filter out all nan items
cnt=0
filtered=[]
rowidx=[]

for i in range(allinfos.shape[0]):
    if np.isnan(allinfos[i,:]).all():
        cnt+=1
    else:
        rowidx.append(i)
        filtered.append(allinfos[i,:])
filtered=np.stack(filtered,0)#1739 left 768 out
# get number of nan for every
cnts=[]
for i in range(filtered.shape[0]):
    cnts.append(np.isnan(filtered[i,:]).sum())
    hist,x=np.histogram(cnts,)
#print(hist)#1246  165 < 5.6

#fixed!
for i in range(filtered.shape[0]):
    if np.isnan(filtered[i,:]).sum()<2:
        for j in range(filtered.shape[1]):
            if np.isnan(filtered[i,j]):
                filtered[i,j]=means[j]

filtered2=[]
rowidx2=[]
cnt=0
for i in range(filtered.shape[0]):
    if np.isnan(filtered[i,:]).sum()>2:
        cnt+=1
    else:
        rowidx2.append(rowidx[i])
        filtered2.append(filtered[i,:])
filtered2=np.stack(filtered2,0)


#find nan cols
# cnts=[]
# for i in range(filtered2.shape[1]):
#     cnts.append(np.isnan(filtered2[:,i]).sum()) 
#     hist,x=np.histogram(cnts,)
#print(hist)#<130

#fixed 2!
filtered_new=[]
colidx=[]
for i in range(filtered2.shape[1]):
    if np.isnan(filtered2[:,i]).sum()<50:
        for j in range(filtered2.shape[0]):
            if np.isnan(filtered2[j,i]):
                filtered2[j,i]=means[i]
        filtered_new.append(filtered2[:,i])
        colidx.append(i)
filtered_new=np.stack(filtered_new,1)
filtered_new=filtered_new-filtered_new.min(0)
filtered_new=filtered_new/filtered_new.max(0)
newdict=dict()
newdict['used_clinical_feature_idx']=colidx
kk=np.array(keys)[np.array(rowidx2)].tolist()
for item in data.keys():
    if item in kk:
        thisone=data[item]
        thisone['clinic_f']=filtered_new[kk.index(item)].tolist()
        newdict[item]=thisone
json.dump(newdict,open('for_dbz_pre/jsons/clinical_onlyx.json','w'))
print(filtered_new.shape)