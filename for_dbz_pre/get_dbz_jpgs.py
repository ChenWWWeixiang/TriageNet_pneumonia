import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os,json
input_path='/mnt/newdisk3/data_for2d'
input_mask='/mnt/newdisk3/seg_for2d'

output_path_slices='/mnt/data9/covid_detector_jpgs/byjb_val'
os.makedirs(output_path_slices,exist_ok=True)
cnt=0
train_list='/mnt/data12/zeroshot_learning/moco/moco/data/val.json'
#train_list2='/mnt/data12/zeroshot_learning/moco/moco/data/more_healthy.json'

#train_list3='/mnt/data12/zeroshot_learning/moco/moco/data/more.json'

train_list=json.load(open(train_list,'r'))
#train_list2=json.load(open(train_list2,'r'))
#train_list3=json.load(open(train_list3,'r'))
info = train_list
cnt=dict()

for idx,name in enumerate(info.keys()):
    #set_name=name.split('/')[-2]
    thisone=info[name]
    #first_name=int(name.split('/')[-1].split('_')[2].split('.')[0])
    set_name=thisone['clsI']
    set2=thisone['clsII']
    input_path = thisone['newpath']
    input_mask = thisone['newpath'].replace('data_for2d','seg_for2d')

    #input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/' + set_name
    volume = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(input_mask)
    M=sitk.GetArrayFromImage(mask)
    M[M>0]=1
    V = sitk.GetArrayFromImage(volume)
    sums = M.sum(1).sum(1)
    idd=np.where(sums>500)
    iddx=np.where(M>0)
    if len(idd[0])==0:
        print(1)
    #M = M[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    #V = V[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    M = M[idd[0],:,:]
    V = V[idd[0],:,:]
    if 'covid' in set_name:
        stride=5
        #continue
    elif 'healthy' in set_name:
        stride=3
    elif 'bacteria' in set_name:
        stride=5
    elif 'fungas' in set_name:
        stride=4
    else:
        stride=1

    for idx, i in enumerate(range(0,V.shape[0],stride)):
        data=V[i,:,:]
        mask=M[i,:,:]
        data[data>500]=500
        data[data<-1200]=-1200#-1200~500
        data = data+1200
        data=data*255.0/1700
        if set2 in cnt.keys():
            cnt[set2]+=1
        else:
            cnt[set2]=0
        data=np.stack([data,data*mask,255*mask],-1)#mask one channel
        data = data.astype(np.uint8)
        #data = cv2.flip(data, 0)
        os.makedirs(os.path.join(output_path_slices,set_name,set2),exist_ok=True)
        cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                                 str(int(i/(V.shape[0])*100))+'.jpg'),data)
print(cnt)
