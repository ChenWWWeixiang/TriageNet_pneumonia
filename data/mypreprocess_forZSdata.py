import SimpleITK as sitk
import numpy as np
import os,glob
import sys
import pydicom,cv2
import argparse
import csv,pandas
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/mnt/data9/covid_detector_jpgs/diff_wc')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    default='/mnt/data11/NCP_CTs/CAP/ZSCAP')
                    #default='/home/cwx/extra/NCP_CTs/NCP_ill')
args = parser.parse_args()

output_path_slices=args.output_path
os.makedirs(output_path_slices,exist_ok=True)
#excel_path='/mnt/data9/zsdx.csv'
#data=pandas.read_csv(excel_path)
#data=[row for row in data]
set_name='CP'
input_path=args.input_path
for i in range(1,2):
    #
    path=input_path
    #path=input_path
    all_id = os.listdir(path)
    for id in all_id:
        cnt=0
        for inner in os.listdir(os.path.join(path,id)):
            for phase in os.listdir(os.path.join(path,id,inner)):
                adicom = os.listdir(os.path.join(path,id,inner,phase))
                adicom = [a for a in adicom if a[0] == 'I']
                adicom = adicom[0]
                # print(os.path.join(root, patient, case, phase, inner, adicom))
                try:
                    case_path=os.path.join(path,id,inner,phase)
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                except:
                    continue

                if image.GetSize()[-1]<=100:
                    continue
                ds = pydicom.read_file(os.path.join(path,id,inner,phase, adicom))
                if not ds['ImageType'].value[-1]=='AXIAL' :
                    continue
                if not 'Chest' in ds['ProtocolName'].value:
                    if not 'CHEST' in ds['ProtocolName'].value:
                        continue
                idx=np.where(data['name']==id)[0]
                id=id.replace('_','-')
                date = '0101'
                if len(idx)>0:
                    age=data['age'][idx[0]]
                    sex=data['sex'][idx[0]]
                    if sex==1:
                        sex='M'
                    else:
                        sex='F'
                else:
                    age=-1
                    sex='M'
                wc = ds['WindowCenter'].value
                ww = ds['WindowWidth'].value
                V = sitk.GetArrayFromImage(image)
                for idx in range(image.GetSize()[-1]):
                    data = V[idx, :, :]
                    data[data > (wc + ww // 2)] = wc + ww // 2
                    data[data < (wc - ww // 2)] = (wc - ww // 2)  # -1200~500
                    data = data - (wc - ww // 2)
                    data = data * 255.0 / ww

                    data = data.astype(np.uint8)
                    os.makedirs(os.path.join(output_path_slices, set_name, str(i) + '_' + str(id), phase),
                                exist_ok=True)
                    cv2.imwrite(os.path.join(output_path_slices, set_name, str(i) + '_' + str(id), phase,
                                             str(idx) + '.jpg'), data)




