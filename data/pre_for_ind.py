import SimpleITK as sitk
import numpy as np
import os,glob,cv2
import sys
import pydicom
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/mnt/data9/covid_detector_jpgs/diff_wc')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    #default='/home/cwx/extra/CAP/CAP')
                    default='/mnt/data11/NCP_CTs/NCP_ill')
args = parser.parse_args()
set_name='NCP'
output_path_slices=args.output_path
os.makedirs(output_path_slices,exist_ok=True)

input_path=args.input_path
reader = sitk.ImageSeriesReader()
#pid_list=open('all_pid.txt','w')
for i in range(3,10):
    #
    path=input_path+str(i)
    #path=input_path
    all_id = os.listdir(path)
    for id in all_id:
        all_phase=os.listdir(os.path.join(path,id))
        num_phase=len(all_phase)
        for phase in all_phase:
            inner=os.listdir(os.path.join(path,id,phase))
            for itemsinnner in inner:
                if itemsinnner == "DICOMDIR" or itemsinnner == 'LOCKFILE' or itemsinnner == 'VERSION':
                    continue
                iinner=os.listdir(os.path.join(path,id,phase,itemsinnner))
                for iinn_item in iinner:
                    if iinn_item=='VERSION':
                        continue
                    try:
                        case_path=os.path.join(path,id,phase,itemsinnner,iinn_item)
                        dicom_names = reader.GetGDCMSeriesFileNames(case_path)
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                    except:
                        continue
                    if image.GetSize()[-1]<=10:
                        continue
                    adicom = os.listdir(os.path.join(path,id,phase,itemsinnner,iinn_item))
                    adicom = [a for a in adicom if a[0] == 'I']
                    adicom = adicom[0]
                    # print(os.path.join(root, patient, case, phase, inner, adicom))
                    try:
                        ds = pydicom.read_file(os.path.join(path, id, phase, itemsinnner, iinn_item, adicom))
                        wc=ds['WindowCenter'].value
                        ww=ds['WindowWidth'].value
                    except:
                        continue
                    V=sitk.GetArrayFromImage(image)
                    for idx in range(image.GetSize()[-1]):
                        data = V[idx, :, :]
                        data[data > (wc+ww//2)] = wc+ww//2
                        data[data < (wc-ww//2)] = (wc-ww//2)  # -1200~500
                        data = data -(wc-ww//2)
                        data = data * 255.0 / ww

                        data = data.astype(np.uint8)
                        os.makedirs(os.path.join(output_path_slices, set_name, str(i)+'_'+str(id), phase), exist_ok=True)
                        cv2.imwrite(os.path.join(output_path_slices, set_name, str(i)+'_'+str(id), phase,
                                                 str(idx) + '.jpg'), data)







