import SimpleITK as sitk
import os,glob,pydicom
root='/home/cwx/healthy_xct'
output='/mnt/data11/dr_ct_2/healthy'
os.makedirs(os.path.join(output,'CT','healthy'),exist_ok=True)
os.makedirs(os.path.join(output,'XR','healthy'),exist_ok=True)
reader = sitk.ImageSeriesReader()
for patient in os.listdir(root):
    cxr=os.path.join(os.path.join(root,patient,'1'))
    ct = os.path.join(os.path.join(root, patient, '2'))
    if not (os.path.exists(cxr) and os.path.exists(ct)):
        continue
    for inner in os.listdir(cxr):
        if os.path.isdir(os.path.join(cxr,inner)):
            for innner in os.listdir(os.path.join(cxr,inner)):
                if os.path.isdir(os.path.join(cxr, inner,innner)):
                    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(cxr, inner,innner))
                    reader.SetFileNames(dicom_names)
                    XRay = reader.Execute()
                    adicom1=glob.glob(os.path.join(cxr, inner, innner,'I*'))
                    ds=pydicom.read_file(os.path.join(cxr, inner, innner, adicom1[0]))
                    date = ds['StudyDate'].value
                    try:
                        age = int(ds['PatientAge'].value[:-1])
                        sex = ds['PatientSex'].value
                    except:
                        age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
                        sex = ds['PatientSex'].value
                    sitk.WriteImage(XRay,os.path.join(output,'XR','healthy','14_'+patient+'_'+date+'_'+str(age)+'_'+sex+'.nii'))
                    break
    for inner in os.listdir(ct):
        if os.path.isdir(os.path.join(ct,inner)):
            for innner in os.listdir(os.path.join(ct,inner)):
                if os.path.isdir(os.path.join(ct, inner,innner)):
                    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(ct, inner,innner))
                    reader.SetFileNames(dicom_names)
                    XRay = reader.Execute()
                    adicom1=glob.glob(os.path.join(ct, inner, innner,'I*'))
                    ds=pydicom.read_file(os.path.join(ct, inner, innner, adicom1[0]))
                    date = ds['StudyDate'].value
                    try:
                        age = int(ds['PatientAge'].value[:-1])
                        sex = ds['PatientSex'].value
                    except:
                        age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
                        sex = ds['PatientSex'].value
                    sitk.WriteImage(XRay,os.path.join(output,'CT','healthy','14_'+patient+'_'+date+'_'+str(age)+'_'+sex+'.nii'))
                    break