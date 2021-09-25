from utils_sitk import *
import json,os
import numpy as np
import SimpleITK as sitk
avp='/mnt/newdisk0/tzm/Data/Airway-Sseg/mean_points.npy'
avp=np.load(avp)
train_old=json.load(open('for_dbz_pre/jsons/train.json','r'))
test_old=json.load(open('for_dbz_pre/jsons/val.json','r')) 
outpath_root='/mnt/newdisk3/resampled1-1-1.5'
target_size=np.array([176,112,176])
allfiles=dict(**train_old,**test_old)
for person in allfiles:
    raw=sitk.ReadImage(allfiles[person]['newpath'])
    size_r=np.array(raw.GetSize())
    spacing_r=np.array(raw.GetSpacing())
    outputsize=(size_r*spacing_r/np.array([1,1,1.5])).astype(np.uint32)
    resampled_data,moving_resampled_ar=get_resampled(raw,outputsize,resampled_spacing=[1,1,1.5]) 
    moving_resampled_ar=moving_resampled_ar.transpose((2,1,0))
    mid=np.array(moving_resampled_ar.shape)/2
    s,e=mid-target_size,mid+target_size
    s,e=s.astype(np.int),e.astype(np.int)
    moving_resampled_ar=moving_resampled_ar[s[0]:e[0],s[1]:e[1],s[2]:e[2]]
    moving_resampled_ar = (moving_resampled_ar - np.mean(moving_resampled_ar)) / np.std(moving_resampled_ar)
    outpath=allfiles[person]['newpath'].replace('/mnt/newdisk3/data_for2d',outpath_root)
    os.makedirs(outpath,exist_ok=True)
    np.save(outpath,moving_resampled_ar)
    a=1