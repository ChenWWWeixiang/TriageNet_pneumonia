import numpy as np
import os
import SimpleITK as sitk
import sklearn.metrics as metric
if True:
    datas=np.load('../re/xray.npy')
    #datas = np.load('../re/xct.npy')
    lesion_root='/home/cwx/extra/covid_project_segs/lesion'
    name=datas[:,0].tolist()
    lesion_seg_names=[na.split('/')[-3]+'/'+na.split('/')[-1].split('.jpg')[0] for na in name]
    ALL=[]
    for idx,one in enumerate(lesion_seg_names):
        if one.split('/')[0]=='CAP':
            continue
            this_path=os.path.join(lesion_root,'cap-zs','cap-zs_1_'+one.split('/')[-1][2:]+'_label.nrrd')
        else:
            idd=int(one.split('/')[-1].split('_')[0])
            if idd==10:
                this_path = os.path.join(lesion_root, 'covid2', 'covid2_' + one.split('/')[-1] + '_label.nrrd')
            else:
                this_path = os.path.join(lesion_root, 'covid', 'covid_' + one.split('/')[-1] + '_label.nrrd')
        if not os.path.exists(this_path):
            print(this_path)
            continue
        data=sitk.ReadImage(this_path)
        data=sitk.GetArrayFromImage(data)
        data=data.sum(1).sum(1)
        valid_infect=np.where(data>50)
        this_pred=np.array(datas[idx,1:],np.float)
        this_pred=np.array([len(valid_infect[0])/data.shape[0]]+this_pred.tolist())
        ALL.append(this_pred)
    ALL=np.array(ALL)
    np.save('infectiou_size_with_xray.npy',ALL)
    #np.save('infectiou_size_with_ct.npy', ALL)
else:
    ALL=np.load('infectiou_size_with_xray.npy')
acc=[]
auc=[]
num=[]
sets=np.array([a for a in ALL if a[0]<0.05 ])
num.append(sets.shape[0])
acc.append(np.mean((sets[:,-2]>0.5)==sets[:,-1]))
#auc.append(metric.roc_auc_score(sets[:,-1],sets[:,-2]))
sets=np.array([a for a in ALL if a[0]>0.05 and a[0]<0.25 ])
num.append(sets.shape[0])
acc.append(np.mean((sets[:,-2]>0.5)==sets[:,-1]))
#auc.append(metric.roc_auc_score(sets[:,-1],sets[:,-2]))
sets=np.array([a for a in ALL if a[0]>0.25 and a[0]<0.5 ])
num.append(sets.shape[0])
acc.append(np.mean((sets[:,-2]>0.5)==sets[:,-1]))
#auc.append(metric.roc_auc_score(sets[:,-1],sets[:,-2]))
sets=np.array([a for a in ALL if a[0]>0.5])
num.append(sets.shape[0])
acc.append(np.mean((sets[:,-2]>0.5)==sets[:,-1]))
#auc.append(metric.roc_auc_score(sets[:,-1],sets[:,-2]))
print('<0.05,0.1-0.25,0.25-0.5,>0.5')
print('number',num)
print('acc',acc)
#print('auc',auc)
print('mean of infectious:',np.mean(ALL[:,0]))