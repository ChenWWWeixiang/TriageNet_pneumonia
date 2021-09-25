import SimpleITK as sitk
import numpy as np
import random,sys
sys.path.append('/mnt/data9/Lipreading-DenseNet3D-master')
from PIL import Image
import cv2,os,json
from for_dbz_pre.utils_sitk import *
from multiprocessing.dummy import  Pool as threadpool
input_path='/mnt/newdisk3/data_for2d'
input_mask='/mnt/newdisk3/seg_for2d'

output_path_slices='/mnt/data9/covid_detector_jpgs/onec_jpg_train'
os.makedirs(output_path_slices,exist_ok=True)
cnt=0
train_list='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/train.json'
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
    if os.path.exists(input_cyst) and False:
        cyst=sitk.ReadImage(input_cyst)
        V = sitk.GetArrayFromImage(volume)
        C = sitk.GetArrayFromImage(cc)
        M=sitk.GetArrayFromImage(mask)
        cyst,CX=get_matched_segs(volume,cyst)
        sums = CX.sum(1).sum(1)
        idd=np.where(sums>500)
        #iddx=np.where(M>0)
    else:
        #cyst=mask
        #lung filter
        M=sitk.GetArrayFromImage(mask)
        
        M[M>0]=1
        V = sitk.GetArrayFromImage(volume)
        C = sitk.GetArrayFromImage(cc)
        #Cyst=sitk.GetArrayFromImage(cyst)
        #Cyst[Cyst>0]=1
        sums = M.sum(1).sum(1)
        idd=np.where(sums>500)
        #iddx=np.where(M>0)
    if len(idd[0])==0:
        print(1)
    M = M[idd[0],:,:]#cyst or lung
    V = V[idd[0],:,:]
    C=C[idd[0],:,:]

    if 'H7N9' in set2:
        aug=5
    elif 'crytococcus' in set2:
        aug=5
    elif 'PCP' in set2:
        aug=3
    elif 'mycoplasma' in set2:
        aug=3    
    else:
        aug=-1
    for idx, i in enumerate(range(0,V.shape[0],stride)):
        data=V[i,:,:]
        #mc_mask=np.zeros((data.shape[0],data.shape[1],14))
        #mc_mask[:,:,0]=M[i,:,:]#lung 0
        mask=M[i,:,:]
        cc=C[i,:,:]
        #for j in range(1,14):#1-13
        #    mc_mask[:,:,j]=cc==j
        data[data>500]=500
        data[data<-1200]=-1200#-1200~500
        data = data+1200
        data=data*255.0/1700
        if set2 in cnt.keys():
            cnt[set2]+=1
        else:
            cnt[set2]=0
        data=np.stack([data,cc*255,255*mask],-1)#mask one channel
        os.makedirs(os.path.join(output_path_slices,set_name,set2),exist_ok=True)
        data = data.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                                 str(int(i/(V.shape[0])*100))+':0.jpg'),data)
       # for j in range(1,15):
        #    t=mc_mask[:,:,j-1]*255
        #    t=t.astype(np.uint8)
        #    cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
        #                         str(int(i/(V.shape[0])*100))+':0:{}.jpg'.format(j)),t)
        
        # #if aug >0:
        # #    for augt in range(aug):
        #         cnt[set2]+=1
        #         ang=random.randint(-30,30)
        #         data_t=data
        #         cv2.rotate(data_t,ang)
        #         cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
        #                          str(int(i/(V.shape[0])*100))+':{}.jpg'.format(augt+1)),data_t)
                # for j in range(1,15):
                #     t=mc_mask[:,:,j-1]*255
                #     t=t.astype(np.uint8)
                #     cv2.rotate(t,ang)
                #     cv2.imwrite(os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                #                         str(int(i/(V.shape[0])*100))+':{}.jpg'.format(augt+1,j)),t)
               # print(augt,'ok for',os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                                       # str(int(i/(V.shape[0])*100))))
        print('ok for',os.path.join(output_path_slices,set_name,set2,thisone['pid']+'_'+str(thisone['age'])+'_'+thisone['gender']+'_'+
                    str(int(i/(V.shape[0])*100))))
pools = threadpool(32)
pools.map(pro,info.keys())
pools.close()
pools.join()
print(cnt)
