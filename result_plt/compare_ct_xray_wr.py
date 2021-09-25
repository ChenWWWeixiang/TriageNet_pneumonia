import numpy as np
import os,cv2,shutil
import SimpleITK as sitk
datas_xray=np.load('../re/xray.npy')
namex_ray=datas_xray[:,0]
datas_ct=np.load('../re/xct.npy')
name_ct=datas_ct[:,0]
namex_ray=[na.split('/')[-3]+'/'+na.split('/')[-1].split('.jpg')[0] for na in namex_ray]
name_ct=[na.split('/')[-2]+'/'+na.split('/')[-1].split('.nii')[0] for na in name_ct]
new_ct_Data=[]
new_ct_Name=[]
for idx,item in enumerate(name_ct):
    if item in namex_ray:
        new_ct_Data.append(datas_ct[idx,1:])
        new_ct_Name.append(item)
datas_ct=np.array(new_ct_Data,np.float)
name_ct=np.array(new_ct_Name)
namex_ray=np.array(namex_ray)

new_xr_Data=[]
new_xr_Name=[]
for idx,item in enumerate(namex_ray):
    if item in name_ct:
        new_xr_Data.append(datas_xray[idx,1:])
        new_xr_Name.append(item)

datas_xray=np.array(new_xr_Data,np.float)
namex_ray=np.array(new_xr_Name)


idx_xray=np.argsort(namex_ray)
idx_ct=np.argsort(name_ct)


valid_num=250

pred1=np.array(datas_xray[:,-2],np.float)[idx_xray]
gt1=np.array(datas_xray[:,-1],np.float)[idx_xray]
error=np.where((pred1>0.5)!=gt1)
xray_error_name=namex_ray[idx_xray][error[0]]
name_ct=name_ct[idx_ct]
namex_ray=namex_ray[idx_xray]

pred2=np.array(datas_ct[:,-2],np.float)[idx_ct]
gt2=np.array(datas_ct[:,-1],np.float)[idx_ct]
error=np.where((pred2>0.986)!=gt2)
ct_error_name=name_ct[error[0]]
p=0.95
pp=pred1*(1-p)+(pred2-0.486)*p
gtgt=gt1
error=np.where((pp>0.5)!=gtgt)
fusion_wrong_name=set(name_ct[error[0]])
#print(len(fusion_wrong_name))
fusion=np.stack([name_ct,1-pp,pp,gtgt],1)
np.save('fusion.npy',fusion)


xray_error_name=set(xray_error_name)
ct_error_name=set(ct_error_name)
a=1
all_ct=set(name_ct).difference(ct_error_name)
all_xr=set(namex_ray).difference(xray_error_name)
all_fusion=set(namex_ray).difference(fusion_wrong_name)
cc_name=all_xr.intersection(all_ct)
print('Error of Fusion:',len(fusion_wrong_name))
print('Error of CXR:',len(xray_error_name),'Error of CT:',len(ct_error_name))
print('Error CXR while correct CT:',len(xray_error_name.difference(ct_error_name)))
print('Error CT while correct CXR:',len(ct_error_name.difference(xray_error_name)))
print('Error both:',len(ct_error_name.intersection(xray_error_name)))
print('Fusion correct both:',len(ct_error_name.intersection(xray_error_name).intersection(all_fusion)))
print('Fusion correct CT:',len(ct_error_name.intersection(all_fusion)))
print('Fusion correct CXR:',len(xray_error_name.intersection(all_fusion)))
print('Fusion disturb CT:',len(all_ct.intersection(fusion_wrong_name)))
#print(ct_error_name.intersection(xray_error_name))
#print(len(namex_ray))
ww='/mnt/data9/output_jpgs_check/ct0_cxr0'
cw='/mnt/data9/output_jpgs_check/ct1_cxr0'
wc='/mnt/data9/output_jpgs_check/ct0_cxr1'
cc='/mnt/data9/output_jpgs_check/ct1_cxr1'
fc='/mnt/data9/output_jpgs_check/fcc'
fx='/mnt/data9/output_jpgs_check/fcr'
fd='/mnt/data9/output_jpgs_check/fdc'
lesion_root='/home/cwx/extra/covid_project_segs/lesion'
def get_img(out_path,set):
    for item in set:
        if item.split('/')[0]=='covid':
            idd = int(item.split('/')[-1].split('_')[0])
            if idd == 10:
                this_path = os.path.join(lesion_root, 'covid2', 'covid2_' + item.split('/')[-1] + '_label.nrrd')
            else:
                this_path = os.path.join(lesion_root, 'covid', 'covid_' + item.split('/')[-1] + '_label.nrrd')
        else :
            this_path = os.path.join(lesion_root, 'cap-zs', 'cap-zs_1_' + item.split('/')[-1][2:] + '_label.nrrd')
        if not os.path.exists(this_path):
            print(this_path)
            continue
        xray_path=os.path.join('/home/tzm/Data/XR_CT_pair/test',item.split('/')[0],'lung_resize',item.split('/')[1]+'.jpg')
        cxr=cv2.imread(xray_path)
        cxr=cv2.resize(cxr,(512,512))
        ct_path=os.path.join('/home/cwx/extra/dr_ct_data/CT',item+'.nii')
        ct=sitk.ReadImage(ct_path)
        ct=sitk.GetArrayFromImage(ct)
        data=sitk.ReadImage(this_path)
        data=sitk.GetArrayFromImage(data)
        if not ct.shape[0]==data.shape[0]:
            continue
        t=data.sum(1).sum(1)
        t=np.where(t>10)
        t=np.mean(t[0]).astype(np.int)
        try:
            ct=ct[t,:,:]
            ct[ct >100] = 100
            ct[ct<-1400]=-1400
            ct=ct+1400
            ct=ct*1.0/1500*255
            I=np.concatenate([cxr[:,:,0],ct],1).astype(np.uint8)
            cv2.imwrite(os.path.join(out_path,item.replace('/','-')+'.jpg'),I)
        except:
            continue
get_img(cw,xray_error_name.difference(ct_error_name))
get_img(wc,ct_error_name.difference(xray_error_name))
get_img(ww,ct_error_name.intersection(xray_error_name))
get_img(cc,cc_name)
#get_img(fc,ct_error_name.intersection(all_fusion))
#get_img(fx,xray_error_name.intersection(all_fusion))
#get_img(fd,all_ct.intersection(fusion_wrong_name))


