import numpy as np
import SimpleITK as sitk
import os,cv2
xray_dir_name=['Healthy/lung_resize','CAP/lung_resize','covid/lung_resize']
mod=['train','test']
xray_root='/home/tzm/Data/XR_CT_pair/'
ct_root='/home/cwx/extra/dr_ct_data/CT/'
seg_root='/mnt/data11/seg_of_XCT/lung'
ct_h_root='/mnt/data11/dr_ct_data/CT/Healthy'
out_path='/mnt/data7/xct_data_npy'
for m in mod:
     for dr_n in xray_dir_name:
         x_root_this=os.path.join(xray_root,m,dr_n)
         all_file=os.listdir(x_root_this)
         if dr_n.split('/')[0]=='Healthy':
             all_cts = [os.path.join(ct_h_root, na.split('.jpg')[0] + '.nii') for na in all_file]
             seg_cts = [
                 os.path.join(seg_root, dr_n.split('/')[0], dr_n.split('/')[0].lower() + '_' + na.split('.jpg')[0] + '.nii') for
                 na in all_file]
         else:
            ct_name=os.path.join(ct_root,dr_n.split('/')[0])
            all_cts=[os.path.join(ct_name,na.split('.jpg')[0]+'.nii') for na in all_file]
            seg_cts=[os.path.join(seg_root,dr_n.split('/')[0],dr_n.split('/')[0]+'_'+na.split('.jpg')[0]+'.nii') for na in all_file]
         a=1
         for a_ct,a_seg,a_xr in zip(all_cts,seg_cts,all_file):
            if os.path.exists(os.path.join(out_path,m,dr_n.split('/')[0],a_xr.split('.jpg')[0]+'.npy')):
                continue
            ctdata=sitk.ReadImage(a_ct)
            ctmask=sitk.ReadImage(a_seg)
            ctdata=sitk.GetArrayFromImage(ctdata)
            ctmask = sitk.GetArrayFromImage(ctmask)
            ctmask[ctmask > 0] = 1
            sums = ctmask.sum(1).sum(1)
            idd = np.where(sums > 500)
            iddx = np.where(ctmask > 0)
            try:
                ctmask = ctmask[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
                ctdata = ctdata[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
                ctdata[ctdata > 500] = 500
                ctdata[ctdata < -1200] = -1200  # -1200~500
                ctdata = ctdata * 255.0 / 1700
                ctdata = ctdata - ctdata.min()
                ctdata=np.stack([ctdata,ctdata*ctmask,ctmask*255]).astype(np.uint8)
                xray=cv2.imread(os.path.join(x_root_this,a_xr))
                alls=np.array([ctdata,xray])
                if not os.path.exists(os.path.join(out_path,m,dr_n.split('/')[0])):
                    os.makedirs(os.path.join(out_path,m,dr_n.split('/')[0]),exist_ok=True)
                np.save(os.path.join(out_path,m,dr_n.split('/')[0],a_xr.split('.jpg')[0]+'.npy'),alls)
            except:
                continue

