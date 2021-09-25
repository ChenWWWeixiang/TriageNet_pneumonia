import SimpleITK as sitk
import cv2,os
import numpy as np
outpath='/mnt/data9/independent_data/covid'
root='/home/cwx/extra/NCP'
for patient in os.listdir(root):
    for scan in os.listdir(os.path.join(root,patient)):
        I=[]
        for slice in os.listdir(os.path.join(root,patient,scan)):
            img=cv2.imread(os.path.join(root,patient,scan,slice))[:,:,1]
            img=img/255.0*2500-1200
            I.append(img)
        I=np.array(I)
        I=sitk.GetImageFromArray(I)
        sitk.WriteImage(I,os.path.join(outpath,patient+'_'+scan+'.nii'))
        print(os.path.join(outpath,patient+'_'+scan+'.nii'))