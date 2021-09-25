import SimpleITK as sitk
import os,glob
import numpy as np
from sklearn.cluster import KMeans

lung_path='/mnt/newdisk3/seg_gz'
lesion_path='/mnt/newdisk2/lesion'
cyst_datas=glob.glob(os.path.join(lesion_path,'*','*','*.seg.nrrd'))
names,rats=[],[]
allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
ish=[item.split('/')[4]=='healthy' for item in cyst_datas]
for item in cyst_datas:
    if item.split('/')[5] =='healthy':
        a=1
    if item.split('/')[5] not in allsubtype:
        continue
    c_data=sitk.GetArrayFromImage(sitk.ReadImage(item))
    try:
        l_data=sitk.GetArrayFromImage(sitk.ReadImage(item.replace('.seg.nrrd','.nii.gz').replace('newdisk2/lesion','newdisk3/seg_gz')))
    except:
        xx=item.split('/')
        y=glob.glob(os.path.join(lung_path,xx[4],xx[5],xx[6].split('-')[0]+'*.nii.gz'))
        try:
            l_data=sitk.GetArrayFromImage(sitk.ReadImage(y[0]))
        except:
            continue
    ratio=np.sum(c_data)*1.0/(np.sum(l_data)+1e-4)
    name=item.split('/')[-1].split('-')[0]
    names.append(name)
    rats.append(ratio)

rats=np.array(rats).reshape(-1,1)
names=np.array(names).reshape(-1,1)
clf = KMeans(n_clusters=2)
clf.fit(rats)
labels = clf.labels_
centers = clf.cluster_centers_
if centers[0]<centers[1]:
    h=1
else:
    h=0
nameh=names[labels==h].tolist()
namel=names[labels==(1-h)].tolist()
print(len(nameh),len(namel))
with open('clsuter.txt','w')as f:
    for item in zip(nameh):
        f.writelines(f'severe\t{item}\n')
    for item in zip(namel):
        f.writelines(f'mild\t{item}\n')


