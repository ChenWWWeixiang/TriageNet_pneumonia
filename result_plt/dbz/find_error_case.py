import numpy as np
import json,shutil,os
allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
f='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/val.json'
data_list=json.load(open(f,'r'))
record='/mnt/data9/Lipreading-DenseNet3D-master/re/temp_mild.npylevel2.npy'
records=np.load(record)
dest='/mnt/newdisk3/checklist'
name=records[:,0]
pre = np.array(records[:, 1:-1], np.float)
gt = np.array(records[:, -1], np.float)
pred=np.argmax(pre,1)
cls_name=[t.split('/')[5] for t in name]
a=1
check_list=['aspergillus','chlamydia','Streptococcus']
for id,item in enumerate(cls_name):
    if item in check_list:
        if not pred[id]==gt[id]: 
            print('report',name[id])
            shutil.copy(name[id],os.path.join(dest,name[id].split('/')[-1]))