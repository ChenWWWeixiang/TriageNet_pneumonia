import SimpleITK as sitk
import numpy as np
import random,sys
sys.path.append('/mnt/data9/Lipreading-DenseNet3D-master')
from PIL import Image
import cv2,os,json
from multiprocessing.dummy import  Pool as threadpool
from for_dbz_pre.utils_sitk import *
input_path='/mnt/newdisk3/data_for2d'
input_mask='/mnt/newdisk3/seg_for2d'

output_path_slices='/mnt/data9/covid_detector_jpgs/croped_filted_val'
os.makedirs(output_path_slices,exist_ok=True)
cnt=0
train_list='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/val.json'
cc_path='/mnt/newdisk2/reg/transform'
cyst_path='/mnt/newdisk2/cyst'
train_list=json.load(open(train_list,'r'))
info = train_list
cnt=dict()

def pro(name):
    #set_name=name.split('/')[-2]
    thisone=info[name]
    #first_name=int(name.split('/')[-1].split('_')[2].split('.')[0])
    set_name=thisone['clsI']
    set2=thisone['clsII']
    input_path = thisone['newpath']
    input_mask = thisone['newpath'].replace('data_for2d','seg_for2d')
    input_cc=thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/reg/transform').replace('.nii','.seg.nii')
    input_cyst=thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/cyst').replace('.nii','_Segmentation.seg.nrrd')
    if not os.path.exists(input_cc):
        return
    #input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/' + set_name
    volume = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(input_mask)
    cc=sitk.ReadImage(input_cc)
    if os.path.exists(input_cyst):
        cyst=sitk.ReadImage(input_cyst)
        V = sitk.GetArrayFromImage(volume)
        C = sitk.GetArrayFromImage(cc)
        M=sitk.GetArrayFromImage(mask)
        cyst,Cy=get_matched_segs(volume,cyst)
        sums = Cy.sum(1).sum(1)
        idd=np.where(sums>100)
        iddx=np.where(M>0)
    else:
        #cyst=mask
        #lung filter
        M=sitk.GetArrayFromImage(mask)
        M[M>0]=1
       
        V = sitk.GetArrayFromImage(volume)
        M=M[:V.shape[0],:,:]
        C = sitk.GetArrayFromImage(cc)
        #Cyst=sitk.GetArrayFromImage(cyst)
        #Cyst[Cyst>0]=1
        sums = M.sum(1).sum(1)
        idd=np.where(sums>500)
        iddx=np.where(M>0)
    if len(idd[0])<3 or M.shape[0]<20:
        print(set_name)
    M = M[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]#cyst or lung
    V = V[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    #C=C[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]

    for idx, i in enumerate(range(0,V.shape[0],1)):
        data=V[i,:,:]
        mask=M[i,:,:]
        #cthis=C[i:i+5,:,:].sum(0)
        data[data>500]=500
        data[data<-1200]=-1200#-1200~500
        data = data+1200
        data=data*255.0/1700
        if set2 in cnt.keys():
            cnt[set2]+=1
        else:
            cnt[set2]=0
        data=np.stack([data,data*mask,255*mask],-1)#mask one channel
        os.makedirs(os.path.join(output_path_slices,set_name,set2),exist_ok=True)
        #mc_mask = mc_mask.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                                 str(int(i/(V.shape[0])*100))+'.jpg'),data)
       
pools = threadpool(4)
pools.map(pro,info.keys())
pools.close()
pools.join()
print(cnt)