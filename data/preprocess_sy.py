import SimpleITK as sitk
import numpy as np
import os,glob,shutil
import sys
import pydicom,cv2
import argparse
import csv,pandas
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/mnt/data9/suyuan_nii')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    default='/mnt/data9/suyuan')
                    #default='/home/cwx/extra/NCP_CTs/NCP_ill')
args = parser.parse_args()

output_path=args.output_path
os.makedirs(output_path,exist_ok=True)

path=args.input_path

all_id = os.listdir(path)
reader = sitk.ImageSeriesReader()
name_dict=dict()

for id in all_id:
    adicom = os.listdir(os.path.join(path, id))
    SID = dict()
    for adicom1 in adicom:
        if os.path.isdir(os.path.join(path, id, adicom1)):
            shutil.move(os.path.join(path, id, adicom1),os.path.join(path, adicom1))
            continue
        ds = pydicom.read_file(os.path.join(path, id, adicom1))
        sid = ds['SeriesInstanceUID'].value
        try:
            if not 'AXIAL' in ds['ImageType'].value:
                continue
        except:
            continue
        if sid in SID.keys():
            continue
        else:
            date = ds['StudyDate'].value
            SID[sid] = date
    try:
        age = int(ds['PatientAge'].value[:-1])
        sex = ds['PatientSex'].value
    except:
        age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
        sex = ds['PatientSex'].value
    case_path=os.path.join(path, id)
    series = reader.GetGDCMSeriesIDs(case_path)
    I=[]
    max_size=0
    thisa=[]
    for _, ase in enumerate(series):
        if not ase in SID.keys():
            continue
        if os.path.exists(os.path.join(output_path,id.split('(')[0]+ '_' + SID[ase] + '_' + str(age) + '_' + sex + '.nii')):
            break
        dicom_names = reader.GetGDCMSeriesFileNames(case_path, ase)
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except:
            continue
        if image.GetSize()[-1]<=10:
            continue
        if image.GetSize()[-1]>max_size:
            I=image
            max_size=image.GetSize()[-1]
            thisa=ase
    if not max_size==0:
        output_name = os.path.join(output_path,id.split('(')[0]+ '_' + SID[thisa] + '_' + str(age) + '_' + sex + '.nii')
        print(output_name)
        sitk.WriteImage(I,output_name)




